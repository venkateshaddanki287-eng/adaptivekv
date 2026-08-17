"""Adaptive KV-cache implementation compatible with Hugging Face Transformers.

Implements token-level KV eviction, position tracking, and dynamic mixed-bit quantization.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache:  # type: ignore[no-redef]
        """Fallback base Cache class."""


from adaptivekv.allocator import AdaptiveBitAllocator, AllocationResult
from adaptivekv.config import (
    AdaptiveKVConfig,
    AllocationConfig,
    QuantizerConfig,
    TokenBudgetConfig,
)
from adaptivekv.controller import TokenBudgetController
from adaptivekv.importance import create_importance_analyzer
from adaptivekv.quantizer import CompressedTensor, GroupQuantizer
from adaptivekv.selector import TokenSelectionResult, TokenSelector

# ── Layer Cache Storage ─────────────────────────────────────────────────────

class LayerKVCache:
    """Storage container for key and value states of a single model layer.

    Supports token-level eviction, logical position tracking, and mixed-bit quantization.
    """

    def __init__(self, layer_idx: int, config: AdaptiveKVConfig) -> None:
        self.layer_idx = layer_idx
        self.config = config

        self.quantizer = GroupQuantizer(config.quantizer)
        self.importance_analyzer = create_importance_analyzer(config.importance)
        self.allocator = AdaptiveBitAllocator(config.allocation)
        self.token_selector = TokenSelector(config.token_budget)
        self.budget_controller = TokenBudgetController(config.token_budget)

        self.compressed_keys: CompressedTensor | None = None
        self.compressed_values: CompressedTensor | None = None
        self.last_allocation: AllocationResult | None = None
        self.last_selection: TokenSelectionResult | None = None

        self.retained_keys: torch.Tensor | None = None
        self.retained_values: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None

        self.tokens_seen: int = 0
        self.tokens_evicted: int = 0
        self.is_compileable: bool = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update layer cache with new token key and value states.

        Args:
            key_states: New key tensor (batch, heads, seq_len, head_dim).
            value_states: New value tensor (batch, heads, seq_len, head_dim).
            attention_weights: Optional attention weights for importance scoring.

        Returns:
            Tuple of (dequantized_keys, dequantized_values) for current active cache.
        """
        device = key_states.device
        step_len = key_states.shape[-2]

        # Construct position sequence for incoming tokens
        if self.positions is None or self.positions.numel() == 0:
            start_pos = 0
        else:
            start_pos = int(self.positions[-1].item()) + 1

        new_positions = torch.arange(
            start_pos, start_pos + step_len, dtype=torch.int64, device=device
        )

        # Retrieve existing active keys & values
        if self.retained_keys is not None and self.retained_values is not None:
            existing_keys = self.retained_keys
            existing_values = self.retained_values
            existing_positions = self.positions
        elif self.compressed_keys is not None and self.compressed_values is not None:
            existing_keys = self.quantizer.dequantize(self.compressed_keys)
            existing_values = self.quantizer.dequantize(self.compressed_values)
            existing_positions = self.positions
        else:
            existing_keys = None
            existing_values = None
            existing_positions = None

        # Concatenate incoming states with existing history
        if existing_keys is None or existing_values is None or existing_positions is None:
            combined_keys = key_states
            combined_values = value_states
            combined_positions = new_positions
        else:
            combined_keys = torch.cat([existing_keys, key_states], dim=-2)
            combined_values = torch.cat([existing_values, value_states], dim=-2)
            combined_positions = torch.cat([existing_positions, new_positions], dim=0)

        self.tokens_seen += step_len
        total_seq_len = combined_keys.shape[-2]

        # Calculate budget and apply token eviction if enabled
        tb_cfg = self.config.token_budget
        budget = self.budget_controller.get_budget(
            total_seq_len,
            max_cache_tokens=tb_cfg.max_cache_tokens,
            keep_ratio=tb_cfg.keep_ratio,
            recent_window=tb_cfg.recent_window,
            sink_tokens=tb_cfg.sink_tokens,
            min_cache_tokens=tb_cfg.min_cache_tokens,
        )

        is_evicting = tb_cfg.enable_token_eviction and total_seq_len > budget
        if is_evicting:
            token_scores = self.importance_analyzer.compute_token_importance(
                combined_keys, combined_values, attention_weights=attention_weights
            )
            selection = self.token_selector.select(
                token_scores,
                budget=budget,
                sink_tokens=tb_cfg.sink_tokens,
                recent_window=tb_cfg.recent_window,
            )
            self.last_selection = selection

            keep_idx = selection.keep_indices
            retained_k = combined_keys[:, :, keep_idx, :]
            retained_v = combined_values[:, :, keep_idx, :]
            retained_pos = combined_positions[keep_idx]
            self.tokens_evicted += selection.num_discarded
        else:
            retained_k = combined_keys
            retained_v = combined_values
            retained_pos = combined_positions
            self.last_selection = None

        self.positions = retained_pos

        # Quantize retained tokens if enabled
        if self.config.enable_quantization:
            if is_evicting or self.compressed_keys is None or self.compressed_values is None:
                # Full quantization path on initial prefill/fill or when eviction re-indexes tokens
                if self.config.enable_adaptive_bits:
                    importance = self.importance_analyzer.compute_importance(
                        retained_k,
                        retained_v,
                        attention_weights=attention_weights,
                        group_size=self.config.quantizer.group_size,
                    )
                    self.last_allocation = self.allocator.allocate(importance)
                    allocations = self.last_allocation.allocations
                else:
                    allocations = None
                    self.last_allocation = None

                self.compressed_keys = self.quantizer.quantize(
                    retained_k, allocations=allocations
                )
                self.compressed_values = self.quantizer.quantize(
                    retained_v, allocations=allocations
                )
            else:
                # Incremental quantization path: quantize ONLY new incoming token(s)
                if self.config.enable_adaptive_bits:
                    importance_new = self.importance_analyzer.compute_importance(
                        key_states,
                        value_states,
                        attention_weights=attention_weights,
                        group_size=self.config.quantizer.group_size,
                    )
                    self.last_allocation = self.allocator.allocate(importance_new)
                    allocations_new = self.last_allocation.allocations
                else:
                    allocations_new = None
                    self.last_allocation = None

                new_ck = self.quantizer.quantize(key_states, allocations=allocations_new)
                new_cv = self.quantizer.quantize(value_states, allocations=allocations_new)

                self.compressed_keys = CompressedTensor.concat([self.compressed_keys, new_ck])
                self.compressed_values = CompressedTensor.concat([self.compressed_values, new_cv])

            self.retained_keys = retained_k
            self.retained_values = retained_v
            return retained_k, retained_v

        # Unquantized path
        self.retained_keys = retained_k
        self.retained_values = retained_v
        self.compressed_keys = None
        self.compressed_values = None
        self.last_allocation = None

        return retained_k, retained_v

    def get_seq_length(self) -> int:
        """Return active retained sequence length."""
        if self.positions is not None:
            return self.positions.numel()
        if self.retained_keys is not None:
            return int(self.retained_keys.shape[-2])
        if self.compressed_keys is not None:
            return int(self.compressed_keys.shape[-2])
        return 0

    @property
    def tokens_currently_cached(self) -> int:
        """Number of currently active tokens in cache."""
        return self.get_seq_length()

    def get_resident_bytes(self) -> int:
        """Return active resident memory consumption for this layer in bytes."""
        if self.config.enable_quantization:
            total = 0
            if self.compressed_keys is not None:
                total += self.compressed_keys.compressed_size_bytes
            if self.compressed_values is not None:
                total += self.compressed_values.compressed_size_bytes
            return total
        else:
            total = 0
            if self.retained_keys is not None:
                total += self.retained_keys.nbytes
            if self.retained_values is not None:
                total += self.retained_values.nbytes
            return total


# ── Top-level AdaptiveKVCache ────────────────────────────────────────────────

class AdaptiveKVCache(Cache):
    """Hugging Face compatible adaptive KV-cache implementation with eviction and quantization.

    Example::

        from adaptivekv import AdaptiveKVCache, AdaptiveKVConfig, TokenBudgetConfig

        cache = AdaptiveKVCache(
            enable_token_eviction=True,
            max_cache_tokens=1024,
            keep_ratio=0.5,
            recent_window=128,
            sink_tokens=4,
        )
    """

    def __init__(
        self,
        config: AdaptiveKVConfig | None = None,
        bits: tuple[int, ...] = (2, 3, 4),
        strategy: str = "threshold",
        memory_budget_ratio: float | None = None,
        group_size: int = 128,
        enable_token_eviction: bool = False,
        max_cache_tokens: int | None = None,
        keep_ratio: float = 1.0,
        recent_window: int = 128,
        sink_tokens: int = 4,
        min_cache_tokens: int = 16,
        enable_quantization: bool = True,
        enable_adaptive_bits: bool = True,
        **kwargs: Any,
    ) -> None:
        with contextlib.suppress(Exception):
            super().__init__()

        if config is not None:
            self.config = config
        else:
            alloc_cfg = AllocationConfig(
                strategy=strategy,
                bits=bits,
                memory_budget_ratio=memory_budget_ratio,
            )
            quant_cfg = QuantizerConfig(group_size=group_size)
            tb_cfg = TokenBudgetConfig(
                enable_token_eviction=enable_token_eviction,
                max_cache_tokens=max_cache_tokens,
                keep_ratio=keep_ratio,
                recent_window=recent_window,
                sink_tokens=sink_tokens,
                min_cache_tokens=min_cache_tokens,
            )
            self.config = AdaptiveKVConfig(
                allocation=alloc_cfg,
                quantizer=quant_cfg,
                token_budget=tb_cfg,
                enable_quantization=enable_quantization,
                enable_adaptive_bits=enable_adaptive_bits,
            )

        self.layers: dict[int, LayerKVCache] = {}  # type: ignore[assignment]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for specified layer index.

        Args:
            key_states: Key states for current step (batch, heads, seq_len, head_dim).
            value_states: Value states for current step.
            layer_idx: Layer index in transformer model.
            cache_kwargs: Optional dict containing 'attention_weights'.

        Returns:
            Tuple of (all_keys, all_values) ready for attention multiplication.
        """
        if layer_idx not in self.layers:
            self.layers[layer_idx] = LayerKVCache(layer_idx, self.config)

        attn_weights = None
        if cache_kwargs is not None and "attention_weights" in cache_kwargs:
            attn_weights = cache_kwargs["attention_weights"]

        return self.layers[layer_idx].update(
            key_states, value_states, attention_weights=attn_weights
        )

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Return active cached sequence length for layer."""
        idx = layer_idx if layer_idx is not None else 0
        if idx not in self.layers:
            return 0
        return self.layers[idx].get_seq_length()

    def get_max_length(self, layer_idx: int | None = None) -> int | None:  # type: ignore[override]
        """Return maximum sequence length."""
        return self.config.token_budget.max_cache_tokens

    def get_mask_sizes(
        self,
        cache_position_or_query_length: int | torch.Tensor = 0,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return active sequence length and offset for causal mask construction."""
        seq_len = self.get_seq_length(layer_idx)
        return seq_len, 0

    def __len__(self) -> int:
        """Return number of cached layers."""
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        """Return layer cache instance for specified layer index."""
        return self.layers[layer_idx]

    @property
    def is_compileable(self) -> bool:
        """Return False as AdaptiveKVCache uses dynamic quantization & eviction structures."""
        return False

    @property
    def tokens_seen(self) -> int:
        """Total cumulative tokens received across all layers."""
        if not self.layers:
            return 0
        return sum(layer.tokens_seen for layer in self.layers.values())

    @property
    def tokens_currently_cached(self) -> int:
        """Total active tokens stored across all layers."""
        if not self.layers:
            return 0
        return sum(layer.tokens_currently_cached for layer in self.layers.values())

    @property
    def tokens_evicted(self) -> int:
        """Total evicted tokens across all layers."""
        if not self.layers:
            return 0
        return sum(layer.tokens_evicted for layer in self.layers.values())

    @property
    def token_retention_ratio(self) -> float:
        """Fraction of total seen tokens currently retained in cache."""
        seen = self.tokens_seen
        if seen == 0:
            return 1.0
        return self.tokens_currently_cached / float(seen)

    def total_compressed_size_bytes(self) -> int:
        """Return total resident memory consumption across all layer caches in bytes."""
        total = 0
        for layer in self.layers.values():
            total += layer.get_resident_bytes()
        return total

    def original_estimated_kv_bytes(self) -> int:
        """Return memory required if all received tokens were kept uncompressed at FP16."""
        total = 0
        for layer in self.layers.values():
            if layer.compressed_keys is not None:
                shape = layer.compressed_keys.shape
                batch = shape[0] if len(shape) >= 1 else 1
                heads = shape[1] if len(shape) >= 2 else 1
                hdim = shape[-1] if len(shape) >= 4 else 64
            elif layer.retained_keys is not None:
                shape = layer.retained_keys.shape
                batch = shape[0] if len(shape) >= 1 else 1
                heads = shape[1] if len(shape) >= 2 else 1
                hdim = shape[-1] if len(shape) >= 4 else 64
            else:
                batch, heads, hdim = 1, 8, 64

            # 2 bytes per FP16 element * 2 (keys + values)
            layer_orig_bytes = layer.tokens_seen * batch * heads * hdim * 2 * 2
            total += layer_orig_bytes
        return total

    def overall_compression_ratio(self) -> float:
        """Return overall memory reduction ratio relative to uncompressed full-precision history."""
        orig_bytes = self.original_estimated_kv_bytes()
        comp_bytes = self.total_compressed_size_bytes()
        if comp_bytes == 0:
            return 0.0
        if orig_bytes == 0:
            # Fallback to ratio calculated over active compressed layers
            for layer in self.layers.values():
                if layer.compressed_keys is not None and layer.compressed_values is not None:
                    orig_bytes += layer.compressed_keys.original_size_bytes
                    orig_bytes += layer.compressed_values.original_size_bytes
                    comp_bytes += layer.compressed_keys.compressed_size_bytes
                    comp_bytes += layer.compressed_values.compressed_size_bytes
            return orig_bytes / comp_bytes if comp_bytes > 0 else 0.0
        return orig_bytes / float(comp_bytes)
