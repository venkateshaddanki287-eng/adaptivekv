"""Tests for token-level importance scoring and TokenSelector."""

from __future__ import annotations

import torch

from adaptivekv.importance import (
    AttentionImportanceAnalyzer,
    MagnitudeImportanceAnalyzer,
    RecencyImportanceAnalyzer,
)
from adaptivekv.selector import TokenSelector, TokenSelectionResult


class TestTokenImportanceScoring:
    """Test token-level importance score computation across analyzers."""

    def test_attention_token_importance(self) -> None:
        analyzer = AttentionImportanceAnalyzer()
        keys = torch.randn(2, 4, 32, 64)
        values = torch.randn(2, 4, 32, 64)
        attn = torch.rand(2, 4, 1, 32)

        scores = analyzer.compute_token_importance(keys, values, attention_weights=attn)
        assert scores.shape == (32,)
        assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)

    def test_magnitude_token_importance(self) -> None:
        analyzer = MagnitudeImportanceAnalyzer()
        keys = torch.randn(1, 8, 50, 64)
        values = torch.randn(1, 8, 50, 64)

        scores = analyzer.compute_token_importance(keys, values)
        assert scores.shape == (50,)
        assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)

    def test_recency_token_importance(self) -> None:
        analyzer = RecencyImportanceAnalyzer()
        keys = torch.randn(1, 4, 20, 32)
        values = torch.randn(1, 4, 20, 32)

        scores = analyzer.compute_token_importance(keys, values)
        assert scores.shape == (20,)
        # Recency gives higher scores to newer tokens
        assert scores[-1] > scores[0]


class TestTokenSelector:
    """Test TokenSelector budget enforcement, ratio selection, and structural protections."""

    def test_selector_topk_budget(self) -> None:
        selector = TokenSelector()
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=torch.float32)

        result: TokenSelectionResult = selector.select(
            scores, budget=4, sink_tokens=0, recent_window=0
        )
        assert result.num_kept == 4
        assert result.num_discarded == 4
        # Retained indices must be sorted ascending
        assert torch.all(result.keep_indices[:-1] <= result.keep_indices[1:])
        # Top 4 highest scores in input: indices 1 (0.9), 3 (0.8), 5 (0.7), 7 (0.6)
        expected = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
        assert torch.equal(result.keep_indices, expected)

    def test_sink_tokens_protection(self) -> None:
        selector = TokenSelector()
        # Initial sink tokens (indices 0, 1) have score 0.0 (lowest importance)
        scores = torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6], dtype=torch.float32)

        result = selector.select(scores, budget=3, sink_tokens=2, recent_window=0)
        # Sink tokens 0 and 1 MUST be protected and retained!
        assert 0 in result.keep_indices.tolist()
        assert 1 in result.keep_indices.tolist()
        assert result.num_kept == 3

    def test_recent_window_protection(self) -> None:
        selector = TokenSelector()
        # Recent tokens (indices 4, 5) have score 0.0
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.0, 0.0], dtype=torch.float32)

        result = selector.select(scores, budget=3, sink_tokens=0, recent_window=2)
        # Recent tokens 4 and 5 MUST be protected and retained!
        assert 4 in result.keep_indices.tolist()
        assert 5 in result.keep_indices.tolist()
        assert result.num_kept == 3

    def test_deterministic_tie_breaking(self) -> None:
        selector = TokenSelector()
        # All equal scores
        scores = torch.ones(10, dtype=torch.float32)

        r1 = selector.select(scores, budget=5, sink_tokens=1, recent_window=1)
        r2 = selector.select(scores, budget=5, sink_tokens=1, recent_window=1)

        assert torch.equal(r1.keep_indices, r2.keep_indices)
        assert r1.num_kept == 5
