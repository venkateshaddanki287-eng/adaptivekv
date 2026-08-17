"""Hugging Face integration adapters for AdaptiveKV.

Provides model validation, compatibility checking, and adapter binding for supported
decoder-only Hugging Face architectures (Llama, Mistral, Qwen2, Gemma, OPT, GPT-NeoX, GPT-2).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from adaptivekv.cache import AdaptiveKVCache
from adaptivekv.config import AdaptiveKVConfig
from adaptivekv.exceptions import UnsupportedModelError

SUPPORTED_MODEL_TYPES: tuple[str, ...] = (
    "llama",
    "mistral",
    "qwen2",
    "gemma",
    "opt",
    "gpt_neox",
    "gpt2",
)
"""Tuple of Hugging Face model architecture identifiers supported by AdaptiveKV."""


class HuggingFaceAdapter:
    """Adapter for connecting AdaptiveKVCache to Hugging Face decoder-only models."""

    def __init__(
        self,
        supported_types: Sequence[str] = SUPPORTED_MODEL_TYPES,
    ) -> None:
        self.supported_types = tuple(supported_types)

    def validate_model(self, model: torch.nn.Module) -> str:
        """Validate whether a Hugging Face model architecture is supported.

        Args:
            model: Hugging Face PreTrainedModel instance.

        Returns:
            The detected model_type string.

        Raises:
            UnsupportedModelError: If model_type is unknown or not supported.
        """
        model_config = getattr(model, "config", None)
        if model_config is None:
            raise UnsupportedModelError("unknown (missing model.config)")

        model_type = getattr(model_config, "model_type", None)
        if model_type is None:
            # Fallback check on class name
            class_name = model.__class__.__name__.lower()
            for st in self.supported_types:
                if st in class_name:
                    return st
            raise UnsupportedModelError("unknown (missing config.model_type)")

        model_type_str = str(model_type).lower()
        if model_type_str not in self.supported_types:
            raise UnsupportedModelError(model_type_str)

        return model_type_str

    def attach_cache(
        self,
        model: torch.nn.Module,
        cache: AdaptiveKVCache | None = None,
        config: AdaptiveKVConfig | None = None,
    ) -> tuple[torch.nn.Module, AdaptiveKVCache]:
        """Attach AdaptiveKVCache to a supported Hugging Face model.

        Args:
            model: Supported Hugging Face decoder-only model instance.
            cache: Pre-instantiated AdaptiveKVCache. Created if None.
            config: Optional AdaptiveKVConfig used if creating cache.

        Returns:
            Tuple of (model, adaptive_kv_cache).
        """
        self.validate_model(model)
        adaptive_cache = cache if cache is not None else AdaptiveKVCache(config=config)
        return model, adaptive_cache


def apply_adaptive_kv(
    model: torch.nn.Module,
    config: AdaptiveKVConfig | None = None,
    bits: tuple[int, ...] = (2, 3, 4),
    strategy: str = "threshold",
    memory_budget_ratio: float | None = None,
    enable_token_eviction: bool = False,
    max_cache_tokens: int | None = None,
    keep_ratio: float = 1.0,
    recent_window: int = 128,
    sink_tokens: int = 4,
    min_cache_tokens: int = 16,
    enable_quantization: bool = True,
    enable_adaptive_bits: bool = True,
) -> tuple[torch.nn.Module, AdaptiveKVCache]:
    """Helper function to validate and attach AdaptiveKVCache to a Hugging Face model.

    Example::

        from adaptivekv import apply_adaptive_kv

        model, cache = apply_adaptive_kv(
            model,
            enable_token_eviction=True,
            max_cache_tokens=1024,
            keep_ratio=0.5,
        )
        outputs = model.generate(**inputs, past_key_values=cache)
    """
    adapter = HuggingFaceAdapter()
    if config is not None:
        cache = AdaptiveKVCache(config=config)
    else:
        cache = AdaptiveKVCache(
            bits=bits,
            strategy=strategy,
            memory_budget_ratio=memory_budget_ratio,
            enable_token_eviction=enable_token_eviction,
            max_cache_tokens=max_cache_tokens,
            keep_ratio=keep_ratio,
            recent_window=recent_window,
            sink_tokens=sink_tokens,
            min_cache_tokens=min_cache_tokens,
            enable_quantization=enable_quantization,
            enable_adaptive_bits=enable_adaptive_bits,
        )
    return adapter.attach_cache(model, cache=cache)
