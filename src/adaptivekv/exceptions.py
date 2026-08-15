"""AdaptiveKV exception hierarchy.

All public exceptions inherit from :class:`AdaptiveKVError` so callers can
catch a single base class when desired.
"""

from __future__ import annotations


class AdaptiveKVError(Exception):
    """Base exception for all AdaptiveKV errors."""


# ── Configuration ───────────────────────────────────────────────────────────

class ConfigurationError(AdaptiveKVError):
    """Raised when an invalid configuration is provided."""


class InvalidBitWidthError(ConfigurationError):
    """Raised when an unsupported quantization bit-width is requested."""

    def __init__(self, bit_width: int, supported: tuple[int, ...] = (2, 3, 4)) -> None:
        self.bit_width = bit_width
        self.supported = supported
        super().__init__(
            f"Unsupported bit-width {bit_width}. Supported: {supported}"
        )


class InvalidStrategyError(ConfigurationError):
    """Raised when an unknown allocation strategy is requested."""

    def __init__(self, strategy: str, supported: tuple[str, ...]) -> None:
        self.strategy = strategy
        self.supported = supported
        super().__init__(
            f"Unknown strategy '{strategy}'. Supported: {supported}"
        )


# ── Quantization ────────────────────────────────────────────────────────────

class QuantizationError(AdaptiveKVError):
    """Raised when quantization or dequantization fails."""


class EmptyTensorError(QuantizationError):
    """Raised when an empty tensor is passed to the quantizer."""


# ── Importance / Allocation ─────────────────────────────────────────────────

class ImportanceError(AdaptiveKVError):
    """Raised when importance scoring fails."""


class AllocationError(AdaptiveKVError):
    """Raised when bit allocation fails (e.g. infeasible budget)."""


class InfeasibleBudgetError(AllocationError):
    """Raised when the memory budget cannot be satisfied."""

    def __init__(self, budget_bits: float, min_bits: float) -> None:
        self.budget_bits = budget_bits
        self.min_bits = min_bits
        super().__init__(
            f"Memory budget ({budget_bits:.1f} bits) is below the minimum "
            f"possible ({min_bits:.1f} bits). Increase budget or reduce cache size."
        )


# ── Cache ───────────────────────────────────────────────────────────────────

class CacheError(AdaptiveKVError):
    """Raised when KV-cache operations fail."""


# ── Integration ─────────────────────────────────────────────────────────────

class IntegrationError(AdaptiveKVError):
    """Raised when Hugging Face / external model integration fails."""


class UnsupportedModelError(IntegrationError):
    """Raised when a model architecture is not supported."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        super().__init__(
            f"Model architecture '{model_type}' is not currently supported by AdaptiveKV. "
            f"See docs for the list of supported architectures."
        )
