"""Tests for token eviction, position tracking, and resident cache memory in AdaptiveKVCache."""

from __future__ import annotations

import torch

from adaptivekv.cache import AdaptiveKVCache
from adaptivekv.config import AdaptiveKVConfig, TokenBudgetConfig


class TestEvictionCache:
    """Test real token eviction, position tracking, and resident memory."""

    def test_eviction_budget_enforced(self) -> None:
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=32,
                recent_window=8,
                sink_tokens=4,
            )
        )
        cache = AdaptiveKVCache(config=cfg)

        # Feed 100 tokens
        keys = torch.randn(1, 4, 100, 64)
        values = torch.randn(1, 4, 100, 64)

        out_k, out_v = cache.update(keys, values, layer_idx=0)

        # Output shape sequence dimension must be exactly budget (32)
        assert out_k.shape == (1, 4, 32, 64)
        assert out_v.shape == (1, 4, 32, 64)
        assert cache.get_seq_length(0) == 32
        assert cache.tokens_seen == 100
        assert cache.tokens_currently_cached == 32
        assert cache.tokens_evicted == 68

    def test_kv_alignment(self) -> None:
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=20,
                recent_window=4,
                sink_tokens=2,
            )
        )
        cache = AdaptiveKVCache(config=cfg)

        # Distinct keys and values
        keys = torch.arange(50, dtype=torch.float32).reshape(1, 1, 50, 1).repeat(1, 4, 1, 64)
        values = keys * 10.0

        out_k, out_v = cache.update(keys, values, layer_idx=0)

        # Keys and values must retain identical token positions!
        assert out_k.shape == out_v.shape
        layer = cache[0]
        assert layer.last_selection is not None
        retained_indices = layer.last_selection.keep_indices

        # Check values match keys * 10
        assert torch.allclose(out_v, out_k * 10.0, atol=1e-3)
        assert layer.positions is not None
        assert torch.equal(layer.positions, retained_indices)

    def test_position_tracking(self) -> None:
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=10,
                recent_window=2,
                sink_tokens=2,
            )
        )
        cache = AdaptiveKVCache(config=cfg)

        keys = torch.randn(1, 2, 25, 32)
        values = torch.randn(1, 2, 25, 32)

        cache.update(keys, values, layer_idx=0)

        layer = cache[0]
        assert layer.positions is not None
        assert layer.positions.numel() == 10
        # First 2 positions are sink tokens (0, 1)
        assert layer.positions[0].item() == 0
        assert layer.positions[1].item() == 1
        # Last 2 positions are recent tokens (23, 24)
        assert layer.positions[-2].item() == 23
        assert layer.positions[-1].item() == 24

    def test_autoregressive_multi_step_eviction(self) -> None:
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=16,
                recent_window=4,
                sink_tokens=2,
            )
        )
        cache = AdaptiveKVCache(config=cfg)

        # Step 1: Prompt phase (20 tokens)
        k1 = torch.randn(1, 4, 20, 32)
        v1 = torch.randn(1, 4, 20, 32)
        out_k1, out_v1 = cache.update(k1, v1, layer_idx=0)
        assert cache.get_seq_length(0) == 16

        # Step 2: Generation step 1 (1 token)
        k2 = torch.randn(1, 4, 1, 32)
        v2 = torch.randn(1, 4, 1, 32)
        out_k2, out_v2 = cache.update(k2, v2, layer_idx=0)
        assert cache.get_seq_length(0) == 16
        assert out_k2.shape[-2] == 16

        # Step 3: Generation step 2 (1 token)
        k3 = torch.randn(1, 4, 1, 32)
        v3 = torch.randn(1, 4, 1, 32)
        out_k3, out_v3 = cache.update(k3, v3, layer_idx=0)
        assert cache.get_seq_length(0) == 16

        assert cache.tokens_seen == 22
        assert cache.tokens_currently_cached == 16

    def test_quantization_and_eviction_combined(self) -> None:
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=32,
            ),
            enable_quantization=True,
            enable_adaptive_bits=True,
        )
        cache = AdaptiveKVCache(config=cfg)

        k = torch.randn(1, 8, 128, 64)
        v = torch.randn(1, 8, 128, 64)

        out_k, out_v = cache.update(k, v, layer_idx=0)

        assert out_k.shape == (1, 8, 32, 64)
        assert cache[0].compressed_keys is not None
        # Compressed shape sequence dim must be 32 (evicted!)
        assert cache[0].compressed_keys.shape == (1, 8, 32, 64)
        assert cache.total_compressed_size_bytes() > 0

    def test_no_permanent_raw_copy_retained(self) -> None:
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(
                enable_token_eviction=True,
                max_cache_tokens=16,
            )
        )
        cache = AdaptiveKVCache(config=cfg)

        # Feed 100 tokens
        k = torch.randn(1, 4, 100, 32)
        v = torch.randn(1, 4, 100, 32)
        cache.update(k, v, layer_idx=0)

        layer = cache[0]
        # Full 100-token raw tensors must NOT exist as resident memory
        assert layer.retained_keys is None or layer.retained_keys.shape[-2] == 16
        if layer.compressed_keys is not None:
            assert layer.compressed_keys.shape[-2] == 16
