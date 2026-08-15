"""Tests for adaptivekv.quantizer — quantization engine and bit-packing."""

from __future__ import annotations

import pytest
import torch

from adaptivekv.exceptions import EmptyTensorError, InvalidBitWidthError
from adaptivekv.quantizer import (
    GroupQuantizer,
    pack_bits,
    unpack_bits,
)

# ── Bit Packing / Unpacking Tests ───────────────────────────────────────────

class TestBitPacking:
    """Test lossless packing and unpacking of uint8 integers."""

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_packing_roundtrip(self, bit_width: int) -> None:
        max_val = (1 << bit_width) - 1
        # Create a test sequence of 100 elements covering all integer levels
        data = torch.tensor([i % (max_val + 1) for i in range(100)], dtype=torch.uint8)

        packed = pack_bits(data, bit_width=bit_width)
        unpacked = unpack_bits(packed, bit_width=bit_width, target_numel=100)

        assert torch.equal(data, unpacked)

    def test_2bit_byte_efficiency(self) -> None:
        data = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
        packed = pack_bits(data, bit_width=2)
        assert packed.numel() == 1
        # Expected byte: 0b11_10_01_00 = 0xE4 = 228
        assert packed.item() == 0b11100100

    def test_4bit_byte_efficiency(self) -> None:
        data = torch.tensor([5, 12], dtype=torch.uint8)
        packed = pack_bits(data, bit_width=4)
        assert packed.numel() == 1
        # Expected byte: (12 << 4) | 5 = 0xC5 = 197
        assert packed.item() == 0xC5

    def test_3bit_byte_efficiency(self) -> None:
        # 8 elements -> exactly 3 bytes
        data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0], dtype=torch.uint8)
        packed = pack_bits(data, bit_width=3)
        assert packed.numel() == 3
        unpacked = unpack_bits(packed, bit_width=3, target_numel=8)
        assert torch.equal(data, unpacked)

    def test_invalid_bit_width(self) -> None:
        data = torch.tensor([1, 2], dtype=torch.uint8)
        with pytest.raises(InvalidBitWidthError):
            pack_bits(data, bit_width=5)
        with pytest.raises(InvalidBitWidthError):
            unpack_bits(data, bit_width=1, target_numel=2)

    def test_empty_tensor_packing(self) -> None:
        empty = torch.empty(0, dtype=torch.uint8)
        packed = pack_bits(empty, bit_width=4)
        assert packed.numel() == 0
        unpacked = unpack_bits(packed, bit_width=4, target_numel=0)
        assert unpacked.numel() == 0


# ── Group Quantizer Tests ───────────────────────────────────────────────────

class TestGroupQuantizer:
    """Test group-wise quantization and dequantization."""

    @pytest.fixture
    def quantizer(self) -> GroupQuantizer:
        return GroupQuantizer()

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    @pytest.mark.parametrize("symmetric", [False, True])
    def test_quantize_dequantize_roundtrip(
        self, quantizer: GroupQuantizer, sample_tensor: torch.Tensor, bit_width: int, symmetric: bool
    ) -> None:
        compressed = quantizer.quantize(
            sample_tensor, bit_width=bit_width, group_size=64, symmetric=symmetric
        )
        assert compressed.bit_width == bit_width
        assert compressed.shape == sample_tensor.shape
        assert compressed.dtype == sample_tensor.dtype

        dequantized = quantizer.dequantize(compressed)
        assert dequantized.shape == sample_tensor.shape
        assert dequantized.dtype == sample_tensor.dtype

        # Check reconstruction error bounds
        diff = torch.abs(sample_tensor - dequantized)
        max_error = torch.max(diff).item()

        # Higher bit-width should yield lower max error
        if bit_width == 4:
            assert max_error < 1.0
        elif bit_width == 3:
            assert max_error < 2.0

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_compression_ratio(
        self, quantizer: GroupQuantizer, large_tensor: torch.Tensor, bit_width: int
    ) -> None:
        compressed = quantizer.quantize(
            large_tensor, bit_width=bit_width, group_size=128
        )
        metrics = quantizer.evaluate(large_tensor, compressed)

        # Baseline FP16 is 16 bits per element.
        # Expect compression ratios: 4-bit ~ 3.5x-4x, 3-bit ~ 4.5x-5x, 2-bit ~ 6x-7x
        if bit_width == 4:
            assert metrics.compression_ratio > 3.0
        elif bit_width == 3:
            assert metrics.compression_ratio > 4.0
        elif bit_width == 2:
            assert metrics.compression_ratio > 5.5

    def test_unaligned_tensor_shape(self, quantizer: GroupQuantizer, rng: torch.Generator) -> None:
        # Shape numel = 127, which is prime and not divisible by group_size 32
        tensor = torch.randn(127, generator=rng, dtype=torch.float16)
        compressed = quantizer.quantize(tensor, bit_width=3, group_size=32)
        dequantized = quantizer.dequantize(compressed)

        assert dequantized.shape == tensor.shape
        assert compressed.padding > 0

    def test_empty_tensor_error(self, quantizer: GroupQuantizer) -> None:
        empty = torch.empty(0, dtype=torch.float16)
        with pytest.raises(EmptyTensorError):
            quantizer.quantize(empty)

    def test_invalid_bit_width_error(self, quantizer: GroupQuantizer, sample_tensor: torch.Tensor) -> None:
        with pytest.raises(InvalidBitWidthError):
            quantizer.quantize(sample_tensor, bit_width=5)

    def test_eval_metrics(self, quantizer: GroupQuantizer, sample_tensor: torch.Tensor) -> None:
        compressed = quantizer.quantize(sample_tensor, bit_width=4, group_size=64)
        metrics = quantizer.evaluate(sample_tensor, compressed)

        assert metrics.mse >= 0.0
        assert metrics.max_abs_error >= 0.0
        assert metrics.original_size_bytes > metrics.compressed_size_bytes
        assert metrics.bits_per_element < 16.0

    def test_mixed_precision_quantization(self, quantizer: GroupQuantizer) -> None:
        tensor = torch.randn(4, 128, dtype=torch.float16)  # 4 groups of 128
        allocations = torch.tensor([2, 4, 3, 2], dtype=torch.int64)

        compressed = quantizer.quantize(tensor, group_size=128, allocations=allocations)
        assert compressed.allocations is not None
        assert torch.equal(compressed.allocations, allocations)

        dequantized = quantizer.dequantize(compressed)
        assert dequantized.shape == tensor.shape
        assert dequantized.dtype == tensor.dtype

        diff = torch.abs(tensor - dequantized)
        assert torch.max(diff).item() < 3.0

    @pytest.mark.cuda
    def test_cuda_support(self, quantizer: GroupQuantizer) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        cuda_tensor = torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.float16)
        compressed = quantizer.quantize(cuda_tensor, bit_width=4)
        assert compressed.packed_data.is_cuda
        dequantized = quantizer.dequantize(compressed)
        assert dequantized.is_cuda
        assert dequantized.shape == cuda_tensor.shape

