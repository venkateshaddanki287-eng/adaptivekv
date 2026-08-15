"""Adaptive KV-cache implementation compatible with Hugging Face Transformers.

Implements adaptive quantization and bit-allocation for LLM key-value caches.
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
)
from adaptivekv.importance import create_importance_analyzer
from adaptivekv.quantizer import CompressedTensor, GroupQuantizer

# ── Layer Cache Storage ─────────────────────────────────────────────────────

class LayerKVCache:
    """Storage container for key and value states of a single model layer."""

    def __init__(self, layer_idx: int, config: AdaptiveKVConfig) -> None:
        self.layer_idx = layer_idx
        self.config = config

        self.quantizer = GroupQuantizer(config.quantizer)
        self.importance_analyzer = create_importance_analyzer(config.importance)
        self.allocator = AdaptiveBitAllocator(config.allocation)

        self.compressed_keys: CompressedTensor | None = None
        self.compressed_values: CompressedTensor | None = None
        self.last_allocation: AllocationResult | None = None

        self._raw_keys: torch.Tensor | None = None
        self._raw_values: torch.Tensor | None = None
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
            Tuple of (all_dequantized_keys, all_dequantized_values).
        """
        # Append to raw accumulators
        if self._raw_keys is None or self._raw_values is None:
            self._raw_keys = key_states
            self._raw_values = value_states
        else:
            self._raw_keys = torch.cat([self._raw_keys, key_states], dim=-2)
            self._raw_values = torch.cat([self._raw_values, value_states], dim=-2)

        # Compute importance and bit allocations
        importance = self.importance_analyzer.compute_importance(
            self._raw_keys,
            self._raw_values,
            attention_weights=attention_weights,
            group_size=self.config.quantizer.group_size,
        )
        self.last_allocation = self.allocator.allocate(importance)

        # Quantize keys and values using per-group allocated bit widths
        allocations = self.last_allocation.allocations

        self.compressed_keys = self.quantizer.quantize(
            self._raw_keys, allocations=allocations
        )
        self.compressed_values = self.quantizer.quantize(
            self._raw_values, allocations=allocations
        )

        # Return full dequantized history for model attention calculation
        deq_keys = self.quantizer.dequantize(self.compressed_keys)
        deq_values = self.quantizer.dequantize(self.compressed_values)

        return deq_keys, deq_values

    def get_seq_length(self) -> int:
        """Return total cached sequence length."""
        if self._raw_keys is None:
            return 0
        return int(self._raw_keys.shape[-2])


# ── Top-level AdaptiveKVCache ────────────────────────────────────────────────

class AdaptiveKVCache(Cache):
    """Hugging Face compatible adaptive KV-cache implementation.

    Example::

        from adaptivekv import AdaptiveKVCache

        cache = AdaptiveKVCache(
            bits=(2, 3, 4),
            strategy="budget",
            memory_budget_ratio=0.25,
        )
    """

    def __init__(
        self,
        config: AdaptiveKVConfig | None = None,
        bits: tuple[int, ...] = (2, 3, 4),
        strategy: str = "threshold",
        memory_budget_ratio: float | None = None,
        group_size: int = 128,
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
            self.config = AdaptiveKVConfig(
                allocation=alloc_cfg,
                quantizer=quant_cfg,
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
        """Return cached sequence length for layer."""
        idx = layer_idx if layer_idx is not None else 0
        if idx not in self.layers:
            return 0
        return self.layers[idx].get_seq_length()

    def get_max_length(self, layer_idx: int | None = None) -> int | None:  # type: ignore[override]
        """Return maximum sequence length (unconstrained by default)."""
        return None

    def get_mask_sizes(
        self,
        cache_position_or_query_length: int | torch.Tensor = 0,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return cached sequence length and offset for causal mask construction."""
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
        """Return False as AdaptiveKVCache uses dynamic quantization structures."""
        return False

    def total_compressed_size_bytes(self) -> int:
        """Return total memory consumption across all layer caches in bytes."""
        total = 0
        for layer in self.layers.values():
            if layer.compressed_keys is not None:
                total += layer.compressed_keys.compressed_size_bytes
            if layer.compressed_values is not None:
                total += layer.compressed_values.compressed_size_bytes
        return total

    def overall_compression_ratio(self) -> float:
        """Return overall compression ratio across all layers."""
        orig_bytes = 0
        comp_bytes = 0
        for layer in self.layers.values():
            if layer.compressed_keys is not None and layer.compressed_values is not None:
                orig_bytes += layer.compressed_keys.original_size_bytes
                orig_bytes += layer.compressed_values.original_size_bytes
                comp_bytes += layer.compressed_keys.compressed_size_bytes
                comp_bytes += layer.compressed_values.compressed_size_bytes

        if comp_bytes == 0:
            return 0.0
        return orig_bytes / comp_bytes
