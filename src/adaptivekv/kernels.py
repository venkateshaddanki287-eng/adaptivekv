"""GPU and Triton kernel dispatch utilities for AdaptiveKV.

Provides runtime detection of CUDA and Triton availability, along with PyTorch
vectorized fallbacks for bit-packing and unpacking operations.
"""

from __future__ import annotations

import importlib.util

import torch

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None


def is_triton_available() -> bool:
    """Return True if Triton compiler backend is available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def is_cuda_available() -> bool:
    """Return True if NVIDIA CUDA hardware acceleration is available."""
    return torch.cuda.is_available()


def get_kernel_backend() -> str:
    """Return current kernel execution backend ("triton", "cuda", or "pytorch_cpu")."""
    if is_triton_available():
        return "triton"
    elif is_cuda_available():
        return "cuda"
    else:
        return "pytorch_cpu"
