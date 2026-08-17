"""Tests for TokenBudgetController."""

from __future__ import annotations

from adaptivekv.config import TokenBudgetConfig
from adaptivekv.controller import TokenBudgetController


class TestTokenBudgetController:
    """Test TokenBudgetController calculation across configurations."""

    def test_fixed_max_cache_tokens(self) -> None:
        cfg = TokenBudgetConfig(enable_token_eviction=True, max_cache_tokens=100)
        controller = TokenBudgetController(cfg)

        assert controller.get_budget(50) == 50
        assert controller.get_budget(100) == 100
        assert controller.get_budget(200) == 100

    def test_keep_ratio_budget(self) -> None:
        cfg = TokenBudgetConfig(
            enable_token_eviction=True,
            keep_ratio=0.5,
            sink_tokens=4,
            recent_window=6,
            min_cache_tokens=10,
        )
        controller = TokenBudgetController(cfg)

        # Total 100 tokens: 10 protected (4 sink + 6 recent), 90 compressible.
        # 50% of 90 = 45. Target budget = 10 + 45 = 55 tokens.
        assert controller.get_budget(100) == 55

    def test_min_cache_floor(self) -> None:
        cfg = TokenBudgetConfig(
            enable_token_eviction=True,
            keep_ratio=0.1,
            sink_tokens=2,
            recent_window=2,
            min_cache_tokens=20,
        )
        controller = TokenBudgetController(cfg)

        # Budget calculation yields less than min_cache_tokens, floor is enforced
        assert controller.get_budget(30) >= 20

    def test_zero_sequence(self) -> None:
        controller = TokenBudgetController()
        assert controller.get_budget(0) == 0
