"""Tests for adaptivekv.metrics — evaluation metrics computation."""

from __future__ import annotations

import pytest
import torch

from adaptivekv.metrics import (
    compute_generation_metrics,
    compute_memory_metrics,
    compute_perplexity,
    compute_quality_metrics,
)


class TestMetrics:
    """Test accuracy, memory, and generation metric computations."""

    def test_identical_tensors_quality(self, sample_tensor: torch.Tensor) -> None:
        metrics = compute_quality_metrics(sample_tensor, sample_tensor)
        assert metrics.mse == pytest.approx(0.0, abs=1e-6)
        assert metrics.max_abs_error == pytest.approx(0.0, abs=1e-6)
        assert metrics.cosine_similarity == pytest.approx(1.0, abs=1e-5)

    def test_noisy_tensors_quality(self, sample_tensor: torch.Tensor, rng: torch.Generator) -> None:
        noise = torch.randn_like(sample_tensor, generator=rng) * 0.1
        noisy = sample_tensor + noise

        metrics = compute_quality_metrics(sample_tensor, noisy)
        assert metrics.mse > 0.0
        assert metrics.max_abs_error > 0.0
        assert 0.0 <= metrics.cosine_similarity <= 1.0

    def test_memory_metrics(self) -> None:
        orig = 1000
        comp = 250
        mem = compute_memory_metrics(orig, comp)
        assert mem.original_bytes == 1000
        assert mem.compressed_bytes == 250
        assert mem.compression_ratio == 4.0
        assert mem.memory_saved_percent == 75.0

    def test_generation_metrics(self) -> None:
        gen = compute_generation_metrics(num_tokens=100, latency_seconds=2.0)
        assert gen.num_tokens == 100
        assert gen.latency_seconds == 2.0
        assert gen.tokens_per_second == 50.0

    def test_perplexity_computation(self) -> None:
        logits = torch.randn(1, 10, 100)
        targets = torch.randint(0, 100, (1, 10))
        ppl = compute_perplexity(logits, targets)
        assert ppl > 1.0
