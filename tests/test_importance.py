"""Tests for adaptivekv.importance — KV-cache importance scoring."""

from __future__ import annotations

import pytest
import torch

from adaptivekv.config import ImportanceConfig
from adaptivekv.exceptions import ImportanceError, InvalidStrategyError
from adaptivekv.importance import (
    AttentionImportanceAnalyzer,
    ImportanceScore,
    MagnitudeImportanceAnalyzer,
    RecencyImportanceAnalyzer,
    create_importance_analyzer,
)


class TestAttentionImportanceAnalyzer:
    """Test AttentionImportanceAnalyzer logic."""

    @pytest.fixture
    def analyzer(self) -> AttentionImportanceAnalyzer:
        return AttentionImportanceAnalyzer()

    def test_explicit_attention_weights(self, analyzer: AttentionImportanceAnalyzer) -> None:
        batch, heads, q_len, kv_seq_len = 1, 4, 16, 32
        head_dim = 64

        keys = torch.randn(batch, heads, kv_seq_len, head_dim)
        values = torch.randn(batch, heads, kv_seq_len, head_dim)

        # Create attention weights where token 10 gets 90% of attention
        attn = torch.zeros(batch, heads, q_len, kv_seq_len)
        attn[:, :, :, 10] = 10.0

        score = analyzer.compute_importance(keys, values, attention_weights=attn, group_size=128)
        assert isinstance(score, ImportanceScore)
        assert score.strategy == "attention"
        assert score.scores.min().item() >= 0.0
        assert score.scores.max().item() <= 1.0

    def test_fallback_without_attention_weights(
        self, analyzer: AttentionImportanceAnalyzer, sample_tensor: torch.Tensor
    ) -> None:
        keys = sample_tensor  # (4, 32, 128)
        values = sample_tensor

        score = analyzer.compute_importance(keys, values, attention_weights=None, group_size=128)
        assert score.scores.numel() > 0
        assert score.scores.min().item() >= 0.0
        assert score.scores.max().item() <= 1.0

    def test_empty_keys_raises_error(self, analyzer: AttentionImportanceAnalyzer) -> None:
        keys = torch.empty(0)
        values = torch.empty(0)
        with pytest.raises(ImportanceError):
            analyzer.compute_importance(keys, values)


class TestMagnitudeImportanceAnalyzer:
    """Test MagnitudeImportanceAnalyzer logic."""

    def test_magnitude_scoring(self, sample_tensor: torch.Tensor) -> None:
        analyzer = MagnitudeImportanceAnalyzer()
        # High magnitude key states
        keys = sample_tensor * 100.0
        values = sample_tensor

        score = analyzer.compute_importance(keys, values, group_size=128)
        assert score.strategy == "magnitude"
        assert score.scores.min().item() >= 0.0
        assert score.scores.max().item() <= 1.0


class TestRecencyImportanceAnalyzer:
    """Test RecencyImportanceAnalyzer logic."""

    def test_recency_ramp(self, sample_tensor: torch.Tensor) -> None:
        analyzer = RecencyImportanceAnalyzer()
        score = analyzer.compute_importance(sample_tensor, sample_tensor, group_size=128)
        assert score.strategy == "recency"
        # First group should be <= last group
        if score.num_groups > 1:
            assert score.scores[0].item() < score.scores[-1].item()


class TestImportanceFactory:
    """Test factory creation and config handling."""

    @pytest.mark.parametrize("strat", ["attention", "magnitude", "recency"])
    def test_factory_creation(self, strat: str) -> None:
        cfg = ImportanceConfig(strategy=strat)
        analyzer = create_importance_analyzer(cfg)
        assert analyzer.config.strategy == strat

    def test_factory_invalid_strategy(self) -> None:
        cfg = ImportanceConfig()
        # Bypass __post_init__ to test factory check
        object.__setattr__(cfg, "strategy", "invalid")
        with pytest.raises(InvalidStrategyError):
            create_importance_analyzer(cfg)
