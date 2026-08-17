"""Regression tests for incremental cache optimization in AdaptiveKVCache."""

from __future__ import annotations

import torch

from adaptivekv.cache import AdaptiveKVCache
from adaptivekv.config import AdaptiveKVConfig, TokenBudgetConfig


def test_incremental_update_correctness() -> None:
    """Verify that incremental cache updating matches direct update behavior."""
    cfg = AdaptiveKVConfig(
        token_budget=TokenBudgetConfig(
            enable_token_eviction=False,
        ),
        enable_quantization=True,
        enable_adaptive_bits=True,
    )
    cache = AdaptiveKVCache(config=cfg)

    # Step 1: Prompt phase (10 tokens)
    k1 = torch.randn(1, 4, 10, 64)
    v1 = torch.randn(1, 4, 10, 64)
    out_k1, out_v1 = cache.update(k1, v1, layer_idx=0)
    assert cache.get_seq_length(0) == 10
    assert out_k1.shape == (1, 4, 10, 64)

    # Step 2: Generation decoding step 1 (1 token)
    k2 = torch.randn(1, 4, 1, 64)
    v2 = torch.randn(1, 4, 1, 64)
    out_k2, out_v2 = cache.update(k2, v2, layer_idx=0)
    assert cache.get_seq_length(0) == 11
    assert out_k2.shape == (1, 4, 11, 64)

    layer = cache[0]
    assert layer.compressed_keys is not None
    # Compressed representation shape sequence dim must reflect full 11 tokens
    assert layer.compressed_keys.shape == (1, 4, 11, 64)


def test_no_dequantization_during_decoding() -> None:
    """Verify that no full-cache dequantization occurs during normal decoding steps."""
    cfg = AdaptiveKVConfig(
        token_budget=TokenBudgetConfig(
            enable_token_eviction=False,
        ),
        enable_quantization=True,
        enable_adaptive_bits=True,
    )
    cache = AdaptiveKVCache(config=cfg)

    # Initial prompt fill
    k_prompt = torch.randn(1, 4, 16, 64)
    v_prompt = torch.randn(1, 4, 16, 64)
    cache.update(k_prompt, v_prompt, layer_idx=0)

    layer = cache[0]
    dequant_call_count = 0

    orig_dequantize = layer.quantizer.dequantize

    def spy_dequantize(*args, **kwargs):
        nonlocal dequant_call_count
        dequant_call_count += 1
        return orig_dequantize(*args, **kwargs)

    layer.quantizer.dequantize = spy_dequantize

    # Run 10 decoding steps
    for _ in range(10):
        k_step = torch.randn(1, 4, 1, 64)
        v_step = torch.randn(1, 4, 1, 64)
        cache.update(k_step, v_step, layer_idx=0)

    # Ensure 0 full-cache dequantization calls occurred during decoding steps!
    assert dequant_call_count == 0, f"Expected 0 dequantization calls, got {dequant_call_count}"


def test_eviction_under_incremental_mode() -> None:
    """Verify that eviction is correctly enforced when token budget capacity is reached."""
    cfg = AdaptiveKVConfig(
        token_budget=TokenBudgetConfig(
            enable_token_eviction=True,
            max_cache_tokens=16,
            recent_window=4,
            sink_tokens=2,
        ),
        enable_quantization=True,
        enable_adaptive_bits=True,
    )
    cache = AdaptiveKVCache(config=cfg)

    # Fill to capacity
    k_prompt = torch.randn(1, 4, 16, 64)
    v_prompt = torch.randn(1, 4, 16, 64)
    cache.update(k_prompt, v_prompt, layer_idx=0)
    assert cache.get_seq_length(0) == 16

    # Adding 1 token triggers eviction to maintain budget
    k_step = torch.randn(1, 4, 1, 64)
    v_step = torch.randn(1, 4, 1, 64)
    out_k, out_v = cache.update(k_step, v_step, layer_idx=0)

    assert cache.get_seq_length(0) == 16
    assert out_k.shape[-2] == 16
    assert cache.tokens_evicted > 0
