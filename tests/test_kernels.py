"""Tests for adaptivekv.kernels — backend detection and GPU utilities."""

from __future__ import annotations

from adaptivekv.kernels import get_kernel_backend, is_cuda_available, is_triton_available


class TestKernels:
    """Test kernel backend inspection functions."""

    def test_backend_detection(self) -> None:
        backend = get_kernel_backend()
        assert backend in ("triton", "cuda", "pytorch_cpu")

    def test_cuda_flag(self) -> None:
        assert isinstance(is_cuda_available(), bool)

    def test_triton_flag(self) -> None:
        assert isinstance(is_triton_available(), bool)
