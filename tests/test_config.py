"""Tests for adaptivekv.config — configuration validation and defaults."""

from __future__ import annotations

import dataclasses

import pytest

from adaptivekv.config import (
    AdaptiveKVConfig,
    AllocationConfig,
    AllocationStrategy,
    ImportanceConfig,
    ImportanceStrategy,
    QuantizerConfig,
)
from adaptivekv.exceptions import (
    ConfigurationError,
    InvalidBitWidthError,
    InvalidStrategyError,
)

# ── QuantizerConfig ─────────────────────────────────────────────────────────


class TestQuantizerConfig:
    """QuantizerConfig validation."""

    def test_defaults(self) -> None:
        cfg = QuantizerConfig()
        assert cfg.bit_width == 4
        assert cfg.group_size == 128
        assert cfg.symmetric is False

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_valid_bit_widths(self, bits: int) -> None:
        cfg = QuantizerConfig(bit_width=bits)
        assert cfg.bit_width == bits

    @pytest.mark.parametrize("bits", [0, 1, 5, 8, 16])
    def test_invalid_bit_widths(self, bits: int) -> None:
        with pytest.raises(InvalidBitWidthError):
            QuantizerConfig(bit_width=bits)

    def test_negative_group_size(self) -> None:
        with pytest.raises(ConfigurationError, match="group_size"):
            QuantizerConfig(group_size=0)

    def test_frozen(self) -> None:
        cfg = QuantizerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.bit_width = 2  # type: ignore[misc]


# ── AllocationConfig ────────────────────────────────────────────────────────


class TestAllocationConfig:
    """AllocationConfig validation."""

    def test_defaults(self) -> None:
        cfg = AllocationConfig()
        assert cfg.strategy == "threshold"
        assert cfg.bits == (2, 3, 4)

    def test_budget_requires_ratio(self) -> None:
        with pytest.raises(ConfigurationError, match="memory_budget_ratio"):
            AllocationConfig(strategy="budget")

    def test_budget_valid(self) -> None:
        cfg = AllocationConfig(
            strategy="budget",
            memory_budget_ratio=0.25,
        )
        assert cfg.memory_budget_ratio == 0.25

    def test_budget_ratio_bounds(self) -> None:
        with pytest.raises(ConfigurationError, match="must be in"):
            AllocationConfig(strategy="budget", memory_budget_ratio=0.0)
        with pytest.raises(ConfigurationError, match="must be in"):
            AllocationConfig(strategy="budget", memory_budget_ratio=1.5)

    def test_invalid_strategy(self) -> None:
        with pytest.raises(InvalidStrategyError):
            AllocationConfig(strategy="nonexistent")

    def test_threshold_count_mismatch(self) -> None:
        with pytest.raises(ConfigurationError, match="thresholds"):
            AllocationConfig(bits=(2, 3, 4), thresholds=(0.5,))

    def test_invalid_bit_in_tuple(self) -> None:
        with pytest.raises(InvalidBitWidthError):
            AllocationConfig(bits=(2, 3, 8))


# ── ImportanceConfig ────────────────────────────────────────────────────────


class TestImportanceConfig:
    """ImportanceConfig validation."""

    def test_defaults(self) -> None:
        cfg = ImportanceConfig()
        assert cfg.strategy == "attention"
        assert cfg.normalize is True

    def test_invalid_strategy(self) -> None:
        with pytest.raises(InvalidStrategyError):
            ImportanceConfig(strategy="entropy")


# ── AdaptiveKVConfig ────────────────────────────────────────────────────────


class TestAdaptiveKVConfig:
    """Top-level AdaptiveKVConfig validation."""

    def test_defaults(self) -> None:
        cfg = AdaptiveKVConfig()
        assert cfg.device == "cpu"
        assert cfg.dtype == "float16"
        assert isinstance(cfg.quantizer, QuantizerConfig)
        assert isinstance(cfg.allocation, AllocationConfig)
        assert isinstance(cfg.importance, ImportanceConfig)

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ConfigurationError, match="dtype"):
            AdaptiveKVConfig(dtype="float64")

    def test_replace_creates_new_instance(self) -> None:
        cfg = AdaptiveKVConfig()
        cfg2 = dataclasses.replace(cfg, device="cuda")
        assert cfg.device == "cpu"
        assert cfg2.device == "cuda"

    def test_nested_override(self) -> None:
        cfg = AdaptiveKVConfig(
            quantizer=QuantizerConfig(bit_width=2, group_size=64),
        )
        assert cfg.quantizer.bit_width == 2
        assert cfg.quantizer.group_size == 64


# ── Enum smoke tests ───────────────────────────────────────────────────────


class TestEnums:

    def test_allocation_strategy_values(self) -> None:
        assert AllocationStrategy.THRESHOLD.value == "threshold"
        assert AllocationStrategy.BUDGET.value == "budget"

    def test_importance_strategy_values(self) -> None:
        assert ImportanceStrategy.ATTENTION.value == "attention"
        assert ImportanceStrategy.MAGNITUDE.value == "magnitude"
        assert ImportanceStrategy.RECENCY.value == "recency"
