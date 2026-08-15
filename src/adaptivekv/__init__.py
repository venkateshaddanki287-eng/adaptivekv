"""AdaptiveKV — Adaptive KV-cache compression for LLM inference.

This package provides dynamic bit-allocation for KV-cache quantization,
enabling better memory-vs-quality tradeoffs compared to fixed-bit schemes.
"""

from __future__ import annotations

__version__ = "0.1.0"

from adaptivekv.allocator import AdaptiveBitAllocator, AllocationResult
from adaptivekv.cache import AdaptiveKVCache, LayerKVCache
from adaptivekv.config import (
    AdaptiveKVConfig,
    AllocationConfig,
    AllocationStrategy,
    ImportanceConfig,
    ImportanceStrategy,
    QuantizerConfig,
)
from adaptivekv.exceptions import (
    AdaptiveKVError,
    AllocationError,
    CacheError,
    ConfigurationError,
    EmptyTensorError,
    ImportanceError,
    InfeasibleBudgetError,
    IntegrationError,
    InvalidBitWidthError,
    InvalidStrategyError,
    QuantizationError,
    UnsupportedModelError,
)
from adaptivekv.importance import (
    AttentionImportanceAnalyzer,
    BaseImportanceAnalyzer,
    HeadImportanceAnalyzer,
    ImportanceScore,
    MagnitudeImportanceAnalyzer,
    RecencyImportanceAnalyzer,
    create_importance_analyzer,
)
from adaptivekv.integration import (
    SUPPORTED_MODEL_TYPES,
    HuggingFaceAdapter,
    apply_adaptive_kv,
)
from adaptivekv.kernels import (
    get_kernel_backend,
    is_cuda_available,
    is_triton_available,
)
from adaptivekv.metrics import (
    EvaluationReport,
    GenerationMetrics,
    MemoryMetrics,
    QualityMetrics,
    compute_generation_metrics,
    compute_memory_metrics,
    compute_perplexity,
    compute_quality_metrics,
)
from adaptivekv.quantizer import (
    BaseQuantizer,
    CompressedTensor,
    GroupQuantizer,
    QuantizationMetrics,
    pack_bits,
    unpack_bits,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "AdaptiveKVConfig",
    "QuantizerConfig",
    "AllocationConfig",
    "AllocationStrategy",
    "ImportanceConfig",
    "ImportanceStrategy",
    # Exceptions
    "AdaptiveKVError",
    "ConfigurationError",
    "InvalidBitWidthError",
    "InvalidStrategyError",
    "QuantizationError",
    "EmptyTensorError",
    "ImportanceError",
    "AllocationError",
    "InfeasibleBudgetError",
    "CacheError",
    "IntegrationError",
    "UnsupportedModelError",
    # Quantizer
    "BaseQuantizer",
    "GroupQuantizer",
    "CompressedTensor",
    "QuantizationMetrics",
    "pack_bits",
    "unpack_bits",
    # Importance
    "BaseImportanceAnalyzer",
    "AttentionImportanceAnalyzer",
    "MagnitudeImportanceAnalyzer",
    "RecencyImportanceAnalyzer",
    "HeadImportanceAnalyzer",
    "ImportanceScore",
    "create_importance_analyzer",
    # Allocator
    "AdaptiveBitAllocator",
    "AllocationResult",
    # Cache
    "AdaptiveKVCache",
    "LayerKVCache",
    # Integration
    "HuggingFaceAdapter",
    "apply_adaptive_kv",
    "SUPPORTED_MODEL_TYPES",
    # Metrics
    "QualityMetrics",
    "MemoryMetrics",
    "GenerationMetrics",
    "EvaluationReport",
    "compute_quality_metrics",
    "compute_memory_metrics",
    "compute_generation_metrics",
    "compute_perplexity",
    # Kernels
    "is_triton_available",
    "is_cuda_available",
    "get_kernel_backend",
]
