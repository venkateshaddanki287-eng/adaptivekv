"""Compression, accuracy, and generation latency metrics for AdaptiveKV.

Provides evaluation utilities for measuring reconstruction quality (MSE, Cosine Similarity),
memory savings, generation latency, throughput, and perplexity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ── Metric Data Structures ──────────────────────────────────────────────────

@dataclass
class QualityMetrics:
    """Tensor reconstruction quality metrics.

    Attributes:
        mse: Mean Squared Error between original and reconstructed tensors.
        max_abs_error: Maximum absolute element-wise error.
        cosine_similarity: Cosine similarity between original and reconstructed tensors.
        snr_db: Signal-to-Noise Ratio in decibels (dB).
    """

    mse: float
    max_abs_error: float
    cosine_similarity: float
    snr_db: float


@dataclass
class MemoryMetrics:
    """Memory consumption and savings metrics.

    Attributes:
        original_bytes: Memory required for FP16 baseline in bytes.
        compressed_bytes: Memory required for compressed cache in bytes.
        compression_ratio: Original bytes / compressed bytes.
        memory_saved_percent: Percentage of memory saved ((1 - compressed/original) * 100).
    """

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    memory_saved_percent: float


@dataclass
class GenerationMetrics:
    """Inference throughput and latency metrics.

    Attributes:
        num_tokens: Total generated tokens.
        latency_seconds: Wall-clock generation time in seconds.
        tokens_per_second: Throughput in tokens per second.
    """

    num_tokens: int
    latency_seconds: float
    tokens_per_second: float


@dataclass
class EvaluationReport:
    """Unified evaluation report combining quality, memory, and generation metrics."""

    quality: QualityMetrics
    memory: MemoryMetrics
    generation: GenerationMetrics | None = None
    perplexity: float | None = None


# ── Metric Computation Functions ───────────────────────────────────────────

def compute_quality_metrics(
    original: torch.Tensor, dequantized: torch.Tensor
) -> QualityMetrics:
    """Compute reconstruction quality metrics between original and dequantized tensors.

    Args:
        original: Floating-point ground truth tensor.
        dequantized: Floating-point reconstructed tensor.

    Returns:
        QualityMetrics dataclass.
    """
    orig_f32 = original.to(torch.float32).reshape(-1)
    deq_f32 = dequantized.to(torch.float32).reshape(-1)

    diff = orig_f32 - deq_f32
    mse = float(torch.mean(diff ** 2).item())
    max_abs = float(torch.max(torch.abs(diff)).item())

    # Cosine Similarity
    orig_norm = torch.norm(orig_f32, p=2)
    deq_norm = torch.norm(deq_f32, p=2)

    if orig_norm > 1e-8 and deq_norm > 1e-8:
        dot = torch.dot(orig_f32, deq_f32)
        cos_sim = float((dot / (orig_norm * deq_norm)).item())
    else:
        cos_sim = 1.0

    # Signal-to-Noise Ratio (SNR in dB)
    signal_power = torch.sum(orig_f32 ** 2)
    noise_power = torch.sum(diff ** 2)

    if noise_power > 1e-12 and signal_power > 1e-12:
        snr_db = float((10.0 * torch.log10(signal_power / noise_power)).item())
    else:
        snr_db = 100.0

    return QualityMetrics(
        mse=mse,
        max_abs_error=max_abs,
        cosine_similarity=cos_sim,
        snr_db=snr_db,
    )


def compute_memory_metrics(
    original_bytes: int, compressed_bytes: int
) -> MemoryMetrics:
    """Compute memory savings metrics given raw byte sizes.

    Args:
        original_bytes: Raw uncompressed size in bytes.
        compressed_bytes: Compressed size in bytes.

    Returns:
        MemoryMetrics dataclass.
    """
    if compressed_bytes == 0:
        ratio = 0.0
        saved = 0.0
    else:
        ratio = original_bytes / float(compressed_bytes)
        saved = (1.0 - float(compressed_bytes) / float(original_bytes)) * 100.0

    return MemoryMetrics(
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        memory_saved_percent=saved,
    )


def compute_generation_metrics(
    num_tokens: int, latency_seconds: float
) -> GenerationMetrics:
    """Compute tokens per second and latency.

    Args:
        num_tokens: Number of generated tokens.
        latency_seconds: Duration in seconds.

    Returns:
        GenerationMetrics dataclass.
    """
    tps = (num_tokens / latency_seconds) if latency_seconds > 0 else 0.0
    return GenerationMetrics(
        num_tokens=num_tokens,
        latency_seconds=latency_seconds,
        tokens_per_second=tps,
    )


def compute_perplexity(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Compute perplexity given model logits and target token IDs.

    Args:
        logits: Logits of shape (batch, seq_len, vocab_size).
        target_ids: Target IDs of shape (batch, seq_len).

    Returns:
        Perplexity value.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    return float(math.exp(loss.item()))
