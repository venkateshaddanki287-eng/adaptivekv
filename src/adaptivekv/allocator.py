"""Adaptive bit-level allocator for KV-cache groups.

Assigns bit widths (2, 3, 4) to cache groups based on importance scores using
threshold-based or budget-constrained optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adaptivekv.config import AllocationConfig, AllocationStrategy
from adaptivekv.exceptions import AllocationError, InfeasibleBudgetError, InvalidStrategyError
from adaptivekv.importance import ImportanceScore

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    """Result of adaptive bit allocation across KV-cache groups.

    Attributes:
        allocations: 1D int64 PyTorch tensor containing assigned bit width per group.
        average_bits: Mean bit width allocated per element.
        estimated_compression_ratio: Theoretical compression ratio relative to FP16.
        strategy: Allocation strategy used ("threshold" or "budget").
    """

    allocations: torch.Tensor
    average_bits: float
    estimated_compression_ratio: float
    strategy: str

    @property
    def num_groups(self) -> int:
        """Total number of allocated groups."""
        return self.allocations.numel()


# ── Adaptive Bit Allocator Implementation ───────────────────────────────────

class AdaptiveBitAllocator:
    """Dynamic bit allocator for KV-cache groups.

    Supports threshold-based categorization and greedy budget-constrained bit allocation.
    """

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()

    def allocate(
        self,
        importance: ImportanceScore,
        config: AllocationConfig | None = None,
    ) -> AllocationResult:
        """Allocate bit levels to cache groups given importance scores.

        Args:
            importance: ImportanceScore dataclass containing 1D normalized scores.
            config: Optional override configuration.

        Returns:
            AllocationResult dataclass containing assigned bit width per group.
        """
        cfg = config or self.config
        scores = importance.scores

        if scores.numel() == 0:
            raise AllocationError("Cannot allocate bits for empty importance scores.")

        if cfg.strategy == AllocationStrategy.THRESHOLD.value:
            return self._allocate_threshold(scores, cfg)
        elif cfg.strategy == AllocationStrategy.BUDGET.value:
            return self._allocate_budget(scores, cfg)
        else:
            raise InvalidStrategyError(
                cfg.strategy, (AllocationStrategy.THRESHOLD.value, AllocationStrategy.BUDGET.value)
            )

    def _allocate_threshold(
        self, scores: torch.Tensor, config: AllocationConfig
    ) -> AllocationResult:
        """Threshold-based allocation: low importance -> low bits, high -> high bits."""
        bits = torch.tensor(config.bits, dtype=torch.int64, device=scores.device)
        thresholds = torch.tensor(
            config.thresholds, dtype=torch.float32, device=scores.device
        )

        # Bucketize scores into threshold intervals
        bucket_indices = torch.bucketize(scores, thresholds)
        allocations = bits[bucket_indices]

        avg_bits = float(torch.mean(allocations.to(torch.float32)).item())
        comp_ratio = 16.0 / max(avg_bits, 1e-4)

        return AllocationResult(
            allocations=allocations,
            average_bits=avg_bits,
            estimated_compression_ratio=comp_ratio,
            strategy=config.strategy,
        )

    def _allocate_budget(
        self, scores: torch.Tensor, config: AllocationConfig
    ) -> AllocationResult:
        """Greedy marginal-gain budget allocation to meet memory_budget_ratio."""
        if config.memory_budget_ratio is None:
            raise AllocationError("memory_budget_ratio is required for budget strategy.")

        target_bits_per_elem = config.memory_budget_ratio * 16.0
        sorted_bits = sorted(config.bits)
        min_bits = sorted_bits[0]

        if target_bits_per_elem < min_bits:
            raise InfeasibleBudgetError(
                budget_bits=target_bits_per_elem, min_bits=float(min_bits)
            )

        num_groups = scores.numel()
        device = scores.device

        # Start all groups at minimum bit level
        current_bits = torch.full((num_groups,), min_bits, dtype=torch.int64, device=device)
        current_total_bits = float(min_bits * num_groups)
        max_allowed_total_bits = target_bits_per_elem * num_groups

        # Step-by-step upgrade path: b_curr -> b_next
        # Quality score for bit level b defined as b / 16.0
        for i in range(len(sorted_bits) - 1):
            b_curr = sorted_bits[i]
            b_next = sorted_bits[i + 1]
            bit_diff = b_next - b_curr

            # Identify groups currently at b_curr
            candidate_mask = current_bits == b_curr
            candidate_indices = torch.nonzero(candidate_mask).reshape(-1)

            if candidate_indices.numel() == 0:
                continue

            # Compute marginal gain for candidate groups: importance * (b_next - b_curr)
            candidate_scores = scores[candidate_indices]
            sorted_gain_order = torch.argsort(candidate_scores, descending=True)

            for idx in sorted_gain_order:
                if current_total_bits + bit_diff <= max_allowed_total_bits:
                    group_idx = candidate_indices[idx]
                    current_bits[group_idx] = b_next
                    current_total_bits += bit_diff
                else:
                    break  # Cannot fit further upgrades within budget

        avg_bits = float(torch.mean(current_bits.to(torch.float32)).item())
        comp_ratio = 16.0 / max(avg_bits, 1e-4)

        return AllocationResult(
            allocations=current_bits,
            average_bits=avg_bits,
            estimated_compression_ratio=comp_ratio,
            strategy=config.strategy,
        )
