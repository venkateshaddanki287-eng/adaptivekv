"""Token selection component for AdaptiveKV.

Ranks and selects KV-cache tokens to retain based on token importance scores,
budget constraints, and protection rules (sink tokens and recent window).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adaptivekv.config import TokenBudgetConfig
from adaptivekv.exceptions import CacheError

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class TokenSelectionResult:
    """Result of token selection.

    Attributes:
        keep_indices: 1D int64 PyTorch tensor of retained token sequence indices (sorted ascending).
        discard_indices: 1D int64 PyTorch tensor of evicted token sequence indices (sorted ascending).
        num_kept: Total tokens retained.
        num_discarded: Total tokens evicted.
        retention_ratio: Fraction of tokens retained.
    """

    keep_indices: torch.Tensor
    discard_indices: torch.Tensor
    num_kept: int
    num_discarded: int
    retention_ratio: float


# ── TokenSelector Implementation ────────────────────────────────────────────

class TokenSelector:
    """Selects KV tokens to retain based on importance scores, budget, and structural protections.

    Supports:
    - Protection for initial prefix / sink tokens (`sink_tokens`)
    - Protection for most recent suffix tokens (`recent_window`)
    - Top-K and ratio-based selection over middle compressible tokens
    - Deterministic tie-breaking and strict ascending index order
    """

    def __init__(self, config: TokenBudgetConfig | None = None) -> None:
        self.config = config or TokenBudgetConfig()

    def select(
        self,
        scores: torch.Tensor,
        budget: int,
        sink_tokens: int | None = None,
        recent_window: int | None = None,
    ) -> TokenSelectionResult:
        """Select tokens to keep based on scores, budget, and protection parameters.

        Args:
            scores: 1D float32 PyTorch tensor of token importance scores of shape (seq_len,).
            budget: Target maximum number of tokens to retain.
            sink_tokens: Overriding number of initial prefix tokens to protect.
            recent_window: Overriding number of recent suffix tokens to protect.

        Returns:
            TokenSelectionResult containing sorted keep_indices and discard_indices.
        """
        if scores.numel() == 0:
            raise CacheError("Cannot select tokens from empty score tensor.")

        # Ensure scores is 1D
        scores_1d = scores.reshape(-1).to(torch.float32)
        seq_len = scores_1d.numel()
        device = scores_1d.device

        sink_cfg = sink_tokens if sink_tokens is not None else self.config.sink_tokens
        recent_cfg = recent_window if recent_window is not None else self.config.recent_window

        sink = max(0, min(sink_cfg, min(seq_len, budget)))
        recent = max(0, min(recent_cfg, min(seq_len - sink, max(0, budget - sink))))

        protected_count = sink + recent

        # If sequence is short or budget covers everything, keep all tokens
        if seq_len <= budget or seq_len <= protected_count:
            keep_indices = torch.arange(seq_len, dtype=torch.int64, device=device)
            discard_indices = torch.empty(0, dtype=torch.int64, device=device)
            return TokenSelectionResult(
                keep_indices=keep_indices,
                discard_indices=discard_indices,
                num_kept=seq_len,
                num_discarded=0,
                retention_ratio=1.0,
            )

        # Identify middle compressible range
        mid_start = sink
        mid_end = seq_len - recent
        mid_len = mid_end - mid_start

        # Target middle tokens to keep
        target_mid_keep = max(0, budget - protected_count)
        target_mid_keep = min(target_mid_keep, mid_len)

        if target_mid_keep >= mid_len:
            kept_mid_indices = torch.arange(mid_start, mid_end, dtype=torch.int64, device=device)
        elif target_mid_keep == 0:
            kept_mid_indices = torch.empty(0, dtype=torch.int64, device=device)
        else:
            mid_scores = scores_1d[mid_start:mid_end]
            # Deterministic tie-breaking: add tiny linear ramp to break exact score ties consistently
            tie_breaker = torch.linspace(
                0.0, 1e-7, steps=mid_len, dtype=torch.float32, device=device
            )
            stable_scores = mid_scores + tie_breaker

            _, topk_rel = torch.topk(stable_scores, k=target_mid_keep, largest=True, sorted=True)
            kept_mid_indices = (mid_start + topk_rel).to(torch.int64)

        prefix_indices = torch.arange(0, sink, dtype=torch.int64, device=device)
        suffix_indices = torch.arange(seq_len - recent, seq_len, dtype=torch.int64, device=device)

        all_keep = torch.cat([prefix_indices, kept_mid_indices, suffix_indices])
        keep_indices, _ = torch.sort(all_keep)

        # Compute discard indices
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[keep_indices] = False
        discard_indices = torch.nonzero(mask).reshape(-1).to(torch.int64)

        num_kept = keep_indices.numel()
        num_discarded = discard_indices.numel()
        retention_ratio = num_kept / float(seq_len)

        return TokenSelectionResult(
            keep_indices=keep_indices,
            discard_indices=discard_indices,
            num_kept=num_kept,
            num_discarded=num_discarded,
            retention_ratio=retention_ratio,
        )
