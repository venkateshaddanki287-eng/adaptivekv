"""Tests for adaptivekv.allocator — adaptive bit allocation."""

from __future__ import annotations

import pytest
import torch

from adaptivekv.allocator import AdaptiveBitAllocator, AllocationResult
from adaptivekv.config import AllocationConfig
from adaptivekv.exceptions import AllocationError, InfeasibleBudgetError
from adaptivekv.importance import ImportanceScore


class TestAdaptiveBitAllocatorThreshold:
    """Test threshold-based allocation."""

    @pytest.fixture
    def allocator(self) -> AdaptiveBitAllocator:
        return AdaptiveBitAllocator()

    def test_threshold_allocation_buckets(self, allocator: AdaptiveBitAllocator) -> None:
        # Scores ranging from 0.0 to 1.0
        scores = torch.tensor([0.1, 0.2, 0.4, 0.5, 0.7, 0.9], dtype=torch.float32)
        importance = ImportanceScore(scores=scores, strategy="attention", group_size=128)

        cfg = AllocationConfig(
            strategy="threshold", bits=(2, 3, 4), thresholds=(0.33, 0.66)
        )
        res = allocator.allocate(importance, config=cfg)

        assert isinstance(res, AllocationResult)
        assert res.strategy == "threshold"
        assert res.allocations.tolist() == [2, 2, 3, 3, 4, 4]
        assert res.average_bits == pytest.approx(3.0)


class TestAdaptiveBitAllocatorBudget:
    """Test greedy budget-constrained allocation."""

    @pytest.fixture
    def allocator(self) -> AdaptiveBitAllocator:
        return AdaptiveBitAllocator()

    def test_budget_allocation_satisfies_constraint(self, allocator: AdaptiveBitAllocator) -> None:
        # 10 groups with varying importance
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        importance = ImportanceScore(scores=scores, strategy="attention", group_size=128)

        # Budget ratio 0.20 -> 0.20 * 16 = 3.2 average bits target
        cfg = AllocationConfig(
            strategy="budget", bits=(2, 3, 4), thresholds=(0.33, 0.66), memory_budget_ratio=0.20
        )
        res = allocator.allocate(importance, config=cfg)

        assert res.strategy == "budget"
        assert res.average_bits <= 3.2 + 1e-5
        # Higher importance groups should have >= bits compared to lower importance groups
        assert res.allocations[-1].item() >= res.allocations[0].item()

    def test_infeasible_budget_raises_error(self, allocator: AdaptiveBitAllocator) -> None:
        scores = torch.tensor([0.5, 0.8])
        importance = ImportanceScore(scores=scores, strategy="attention", group_size=128)

        # 0.05 * 16 = 0.8 average bits, which is below min_bits 2
        cfg = AllocationConfig(
            strategy="budget", bits=(2, 3, 4), thresholds=(0.33, 0.66), memory_budget_ratio=0.05
        )
        with pytest.raises(InfeasibleBudgetError):
            allocator.allocate(importance, config=cfg)

    def test_empty_scores_raises_error(self, allocator: AdaptiveBitAllocator) -> None:
        importance = ImportanceScore(scores=torch.empty(0), strategy="attention", group_size=128)
        with pytest.raises(AllocationError):
            allocator.allocate(importance)
