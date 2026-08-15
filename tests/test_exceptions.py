"""Tests for adaptivekv.exceptions — exception hierarchy smoke tests."""

from __future__ import annotations

import pytest

from adaptivekv.exceptions import (
    AdaptiveKVError,
    AllocationError,
    CacheError,
    ConfigurationError,
    EmptyTensorError,
    InfeasibleBudgetError,
    IntegrationError,
    InvalidBitWidthError,
    InvalidStrategyError,
    QuantizationError,
    UnsupportedModelError,
)


class TestExceptionHierarchy:
    """All exceptions should be catchable via AdaptiveKVError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            InvalidBitWidthError,
            InvalidStrategyError,
            QuantizationError,
            EmptyTensorError,
            AllocationError,
            InfeasibleBudgetError,
            CacheError,
            IntegrationError,
            UnsupportedModelError,
        ],
    )
    def test_inherits_base(self, exc_class: type) -> None:
        assert issubclass(exc_class, AdaptiveKVError)


class TestExceptionMessages:
    """Exceptions carry useful diagnostic info."""

    def test_invalid_bit_width(self) -> None:
        exc = InvalidBitWidthError(8)
        assert "8" in str(exc)
        assert exc.bit_width == 8
        assert exc.supported == (2, 3, 4)

    def test_invalid_strategy(self) -> None:
        exc = InvalidStrategyError("foo", ("bar", "baz"))
        assert "foo" in str(exc)
        assert exc.strategy == "foo"
        assert exc.supported == ("bar", "baz")

    def test_infeasible_budget(self) -> None:
        exc = InfeasibleBudgetError(budget_bits=100.0, min_bits=200.0)
        assert "100.0" in str(exc)
        assert "200.0" in str(exc)

    def test_unsupported_model(self) -> None:
        exc = UnsupportedModelError("bloom")
        assert "bloom" in str(exc)
        assert exc.model_type == "bloom"
