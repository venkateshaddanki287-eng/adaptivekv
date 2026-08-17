"""AdaptiveKV configuration system.

Provides typed, validated configuration via :class:`AdaptiveKVConfig` and
sub-configs.  All public configs are frozen dataclasses — create new
instances via :func:`dataclasses.replace` when you need to change a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from adaptivekv.exceptions import (
    ConfigurationError,
    InvalidBitWidthError,
    InvalidStrategyError,
)

# ── Constants ───────────────────────────────────────────────────────────────

SUPPORTED_BIT_WIDTHS: tuple[int, ...] = (2, 3, 4)
"""Quantization bit-widths supported out of the box."""

SUPPORTED_ALLOCATION_STRATEGIES: tuple[str, ...] = ("threshold", "budget")
"""Bit-allocation strategy identifiers."""

SUPPORTED_IMPORTANCE_STRATEGIES: tuple[str, ...] = (
    "attention",
    "magnitude",
    "recency",
)
"""Importance-scoring strategy identifiers (only 'attention' implemented in Phase 3)."""

DEFAULT_GROUP_SIZE: int = 128
"""Default number of elements per quantization group."""


# ── Enums ───────────────────────────────────────────────────────────────────

class AllocationStrategy(str, Enum):
    """Strategy used by the :class:`AdaptiveBitAllocator`."""

    THRESHOLD = "threshold"
    BUDGET = "budget"


class ImportanceStrategy(str, Enum):
    """Strategy used by the :class:`ImportanceAnalyzer`."""

    ATTENTION = "attention"
    MAGNITUDE = "magnitude"
    RECENCY = "recency"


# ── Sub-configs ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuantizerConfig:
    """Configuration for the quantization engine.

    Attributes:
        bit_width: Number of bits for uniform quantization (2, 3, or 4).
        group_size: Number of contiguous elements that share a scale/zero-point.
            Larger groups → smaller overhead, coarser quantization.
        symmetric: If ``True``, use symmetric quantization around zero.
    """

    bit_width: int = 4
    group_size: int = DEFAULT_GROUP_SIZE
    symmetric: bool = False

    def __post_init__(self) -> None:
        if self.bit_width not in SUPPORTED_BIT_WIDTHS:
            raise InvalidBitWidthError(self.bit_width, SUPPORTED_BIT_WIDTHS)
        if self.group_size < 1:
            raise ConfigurationError(
                f"group_size must be >= 1, got {self.group_size}"
            )


@dataclass(frozen=True)
class AllocationConfig:
    """Configuration for the adaptive bit allocator.

    Attributes:
        strategy: ``"threshold"`` or ``"budget"``.
        bits: Tuple of available bit-widths, sorted ascending.
        thresholds: Importance thresholds separating bit-level buckets.
            Length must be ``len(bits) - 1``.  Only used when
            ``strategy="threshold"``.
        memory_budget_ratio: Target memory as a fraction of the FP16 baseline
            (e.g. 0.25 = 25 %).  Only used when ``strategy="budget"``.
    """

    strategy: str = AllocationStrategy.THRESHOLD.value
    bits: tuple[int, ...] = (2, 3, 4)
    thresholds: tuple[float, ...] = (0.33, 0.66)
    memory_budget_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_ALLOCATION_STRATEGIES:
            raise InvalidStrategyError(
                self.strategy, SUPPORTED_ALLOCATION_STRATEGIES
            )
        for b in self.bits:
            if b not in SUPPORTED_BIT_WIDTHS:
                raise InvalidBitWidthError(b, SUPPORTED_BIT_WIDTHS)
        if len(self.thresholds) != len(self.bits) - 1:
            raise ConfigurationError(
                f"Expected {len(self.bits) - 1} thresholds for "
                f"{len(self.bits)} bit levels, got {len(self.thresholds)}"
            )
        if self.strategy == AllocationStrategy.BUDGET.value:
            if self.memory_budget_ratio is None:
                raise ConfigurationError(
                    "memory_budget_ratio is required when strategy='budget'"
                )
            if not 0.0 < self.memory_budget_ratio <= 1.0:
                raise ConfigurationError(
                    f"memory_budget_ratio must be in (0, 1], "
                    f"got {self.memory_budget_ratio}"
                )


@dataclass(frozen=True)
class ImportanceConfig:
    """Configuration for the importance analyzer.

    Attributes:
        strategy: Scoring strategy identifier.
        normalize: Whether to min-max normalize scores to [0, 1].
    """

    strategy: str = ImportanceStrategy.ATTENTION.value
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_IMPORTANCE_STRATEGIES:
            raise InvalidStrategyError(
                self.strategy, SUPPORTED_IMPORTANCE_STRATEGIES
            )


@dataclass(frozen=True)
class TokenBudgetConfig:
    """Configuration for token-level cache budget control and eviction.

    Attributes:
        enable_token_eviction: Whether to prune unwanted KV tokens when over budget.
        max_cache_tokens: Maximum active cached tokens per layer. None = unconstrained.
        keep_ratio: Fraction of compressible historical tokens to retain in [0.0, 1.0].
        recent_window: Number of most recent tokens protected from eviction.
        sink_tokens: Number of initial prefix tokens protected from eviction.
        min_cache_tokens: Minimum total cached tokens guaranteed per layer.
    """

    enable_token_eviction: bool = False
    max_cache_tokens: int | None = None
    keep_ratio: float = 1.0
    recent_window: int = 128
    sink_tokens: int = 4
    min_cache_tokens: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.keep_ratio <= 1.0:
            raise ConfigurationError(
                f"keep_ratio must be in [0.0, 1.0], got {self.keep_ratio}"
            )
        if self.recent_window < 0:
            raise ConfigurationError(
                f"recent_window must be >= 0, got {self.recent_window}"
            )
        if self.sink_tokens < 0:
            raise ConfigurationError(
                f"sink_tokens must be >= 0, got {self.sink_tokens}"
            )
        if self.min_cache_tokens < 0:
            raise ConfigurationError(
                f"min_cache_tokens must be >= 0, got {self.min_cache_tokens}"
            )
        if self.max_cache_tokens is not None and self.max_cache_tokens < 1:
            raise ConfigurationError(
                f"max_cache_tokens must be >= 1 or None, got {self.max_cache_tokens}"
            )


# ── Top-level config ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdaptiveKVConfig:
    """Top-level configuration for AdaptiveKV.

    Example::

        from adaptivekv.config import AdaptiveKVConfig, TokenBudgetConfig

        cfg = AdaptiveKVConfig()           # all defaults
        cfg = AdaptiveKVConfig(
            token_budget=TokenBudgetConfig(enable_token_eviction=True, max_cache_tokens=1024),
            quantizer=QuantizerConfig(bit_width=3),
            allocation=AllocationConfig(strategy="budget", memory_budget_ratio=0.3),
        )

    Attributes:
        quantizer: Default quantizer settings.
        allocation: Adaptive bit-allocation settings.
        importance: Importance-scoring settings.
        token_budget: Token-level eviction and budget settings.
        enable_quantization: Whether quantization is enabled.
        enable_adaptive_bits: Whether adaptive bit allocation is enabled.
        device: Torch device string (``"cpu"``, ``"cuda"``, etc.).
        dtype: Working dtype for dequantized values.
    """

    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    enable_quantization: bool = True
    enable_adaptive_bits: bool = True
    device: str = "cpu"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        valid_dtypes = ("float16", "bfloat16", "float32")
        if self.dtype not in valid_dtypes:
            raise ConfigurationError(
                f"Unsupported dtype '{self.dtype}'. Choose from {valid_dtypes}"
            )

