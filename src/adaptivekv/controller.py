"""Token budget controller for AdaptiveKV.

Calculates dynamic active token retention targets given sequence length,
fixed max cache bounds, ratio constraints, and protection settings.
"""

from __future__ import annotations

from adaptivekv.config import TokenBudgetConfig


class TokenBudgetController:
    """Calculates active target token budget for KV-cache retention.

    Enforces fixed max limits, ratio-based retention, structural token protection,
    and min/max boundary constraints.
    """

    def __init__(self, config: TokenBudgetConfig | None = None) -> None:
        self.config = config or TokenBudgetConfig()

    def get_budget(
        self,
        seq_len: int,
        max_cache_tokens: int | None = None,
        keep_ratio: float | None = None,
        recent_window: int | None = None,
        sink_tokens: int | None = None,
        min_cache_tokens: int | None = None,
    ) -> int:
        """Calculate active target token budget given total sequence length.

        Args:
            seq_len: Current total sequence length.
            max_cache_tokens: Optional override for fixed max cache limit.
            keep_ratio: Optional override for fraction of compressible tokens to retain.
            recent_window: Optional override for recent protected window size.
            sink_tokens: Optional override for sink protected window size.
            min_cache_tokens: Optional override for minimum cache floor.

        Returns:
            Integer target count of tokens that should remain in cache.
        """
        if seq_len <= 0:
            return 0

        max_tokens = max_cache_tokens if max_cache_tokens is not None else self.config.max_cache_tokens
        ratio = keep_ratio if keep_ratio is not None else self.config.keep_ratio
        recent_cfg = recent_window if recent_window is not None else self.config.recent_window
        sink_cfg = sink_tokens if sink_tokens is not None else self.config.sink_tokens
        min_tokens = min_cache_tokens if min_cache_tokens is not None else self.config.min_cache_tokens

        cap = max_tokens if max_tokens is not None else seq_len

        sink = max(0, min(sink_cfg, min(seq_len, cap)))
        recent = max(0, min(recent_cfg, min(seq_len - sink, max(0, cap - sink))))

        protected_tokens = sink + recent
        compressible_tokens = max(0, seq_len - protected_tokens)

        # Ratio-based budget calculation
        retained_compressible = int(round(compressible_tokens * ratio))
        budget = protected_tokens + retained_compressible

        # Fixed max cache constraint
        if max_tokens is not None:
            budget = min(budget, max_tokens)

        # Enforce bounds: cannot exceed total seq_len
        budget = min(seq_len, budget)
        floor = min(seq_len, max(min_tokens, protected_tokens))
        if max_tokens is not None:
            floor = min(floor, max_tokens)

        budget = max(floor, budget)
        return budget
