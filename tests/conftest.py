"""Shared pytest fixtures and configuration for AdaptiveKV tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local src directory takes precedence over installed site-packages
SRC_DIR = str(Path(__file__).parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest
import torch


# ── Device helpers ──────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    """Check CUDA availability without importing heavy backends early."""
    return torch.cuda.is_available()


# ── Marks ───────────────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: requires NVIDIA CUDA GPU")
    config.addinivalue_line("markers", "slow: long-running test")
    config.addinivalue_line("markers", "integration: requires model download")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-skip CUDA tests when no GPU is available."""
    if _cuda_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def device() -> str:
    """Return the best available device string."""
    return "cuda" if _cuda_available() else "cpu"


@pytest.fixture
def rng() -> torch.Generator:
    """Return a seeded CPU random generator for reproducibility."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    return g


@pytest.fixture
def sample_tensor(rng: torch.Generator) -> torch.Tensor:
    """A small (4, 32, 128) FP16 tensor simulating a KV-cache slice.

    Shape semantics: (num_heads, seq_len, head_dim).
    """
    return torch.randn(4, 32, 128, generator=rng, dtype=torch.float16)


@pytest.fixture
def large_tensor(rng: torch.Generator) -> torch.Tensor:
    """A larger (8, 256, 128) FP16 tensor for stress / benchmark tests."""
    return torch.randn(8, 256, 128, generator=rng, dtype=torch.float16)
