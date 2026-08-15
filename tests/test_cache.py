"""Tests for adaptivekv.cache — AdaptiveKVCache behavior."""

from __future__ import annotations

import torch

from adaptivekv.cache import AdaptiveKVCache


class TestAdaptiveKVCache:
    """Test AdaptiveKVCache update and memory reporting."""

    def test_cache_init(self) -> None:
        cache = AdaptiveKVCache(bits=(2, 3, 4), strategy="threshold")
        assert cache.get_seq_length(0) == 0
        assert cache.total_compressed_size_bytes() == 0
        assert cache.overall_compression_ratio() == 0.0

    def test_single_layer_update(self, sample_tensor: torch.Tensor) -> None:
        cache = AdaptiveKVCache(bits=(2, 3, 4), strategy="threshold")
        keys = sample_tensor  # (4, 32, 128)
        values = sample_tensor

        out_keys, out_values = cache.update(keys, values, layer_idx=0)
        assert out_keys.shape == keys.shape
        assert out_values.shape == values.shape
        assert cache.get_seq_length(0) == 32
        assert cache.total_compressed_size_bytes() > 0
        assert cache.overall_compression_ratio() > 1.0

    def test_autoregressive_multi_step_update(self, rng: torch.Generator) -> None:
        cache = AdaptiveKVCache(bits=(2, 3, 4), strategy="threshold")

        # Step 1: Prompt phase (seq_len = 16)
        k1 = torch.randn(1, 8, 16, 64, generator=rng, dtype=torch.float16)
        v1 = torch.randn(1, 8, 16, 64, generator=rng, dtype=torch.float16)
        out_k1, out_v1 = cache.update(k1, v1, layer_idx=0)
        assert out_k1.shape == (1, 8, 16, 64)
        assert cache.get_seq_length(0) == 16

        # Step 2: Generation step 1 (seq_len = 1)
        k2 = torch.randn(1, 8, 1, 64, generator=rng, dtype=torch.float16)
        v2 = torch.randn(1, 8, 1, 64, generator=rng, dtype=torch.float16)
        out_k2, out_v2 = cache.update(k2, v2, layer_idx=0)
        assert out_k2.shape == (1, 8, 17, 64)
        assert cache.get_seq_length(0) == 17

    def test_multi_layer_cache(self, sample_tensor: torch.Tensor) -> None:
        cache = AdaptiveKVCache(bits=(2, 3, 4), strategy="threshold")
        cache.update(sample_tensor, sample_tensor, layer_idx=0)
        cache.update(sample_tensor, sample_tensor, layer_idx=1)

        assert cache.get_seq_length(0) == 32
        assert cache.get_seq_length(1) == 32
        assert len(cache.layers) == 2
        assert cache.total_compressed_size_bytes() > 0
