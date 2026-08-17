"""Quantization engine for KV-cache tensors.

Provides group-wise uniform quantization and bit-packing/unpacking for
2-bit, 3-bit, and 4-bit precision.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from adaptivekv.config import QuantizerConfig
from adaptivekv.exceptions import EmptyTensorError, InvalidBitWidthError

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class CompressedTensor:
    """Container for a quantized and bit-packed tensor.

    Attributes:
        packed_data: Packed uint8 tensor holding 2-bit, 3-bit, or 4-bit values.
        scales: Scale factors per group, shape ``(num_groups, 1)``.
        zero_points: Zero-point offsets per group, shape ``(num_groups, 1)``.
        shape: Original uncompressed tensor shape.
        bit_width: Quantization precision (2, 3, or 4), or average bit width if mixed.
        group_size: Number of elements per quantization group.
        symmetric: Whether symmetric quantization was used.
        dtype: Original floating-point dtype.
        padding: Number of padding elements added to align with group_size and pack_size.
        allocations: Optional 1D int64 tensor containing assigned bit width per group.
    """

    packed_data: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    shape: torch.Size
    bit_width: int
    group_size: int
    symmetric: bool
    dtype: torch.dtype
    padding: int = 0
    allocations: torch.Tensor | None = None

    @property
    def original_numel(self) -> int:
        """Total number of elements in original tensor."""
        return self.shape.numel()

    @property
    def original_size_bytes(self) -> int:
        """Size of original floating-point tensor in bytes."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return self.original_numel * element_size

    @property
    def compressed_size_bytes(self) -> int:
        """Size of compressed representation in bytes (packed data + scales + zero-points + allocations metadata)."""
        alloc_bytes = self.allocations.nbytes if self.allocations is not None else 0
        return (
            self.packed_data.nbytes
            + self.scales.nbytes
            + self.zero_points.nbytes
            + alloc_bytes
        )

    @property
    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size."""
        comp_bytes = self.compressed_size_bytes
        if comp_bytes == 0:
            return 0.0
        return self.original_size_bytes / comp_bytes

    @property
    def bits_per_element(self) -> float:
        """Effective bits per element including overhead."""
        if self.original_numel == 0:
            return 0.0
        return (self.compressed_size_bytes * 8.0) / self.original_numel

    @classmethod
    def concat(cls, tensors: list[CompressedTensor]) -> CompressedTensor:
        """Concatenate multiple CompressedTensor instances along the sequence dimension (-2)."""
        if not tensors:
            raise EmptyTensorError("Cannot concatenate empty list of CompressedTensor instances.")
        if len(tensors) == 1:
            return tensors[0]

        first = tensors[0]
        packed_list = [t.packed_data for t in tensors]
        scales_list = [t.scales for t in tensors]
        zp_list = [t.zero_points for t in tensors]

        has_allocations = any(t.allocations is not None for t in tensors)
        if has_allocations:
            alloc_list = []
            for t in tensors:
                if t.allocations is not None:
                    alloc_list.append(t.allocations)
                else:
                    num_g = t.scales.shape[0]
                    alloc_list.append(torch.full((num_g,), t.bit_width, dtype=torch.int64, device=t.packed_data.device))
            concat_alloc = torch.cat(alloc_list, dim=0)
            avg_bw = int(round(concat_alloc.float().mean().item()))
        else:
            concat_alloc = None
            avg_bw = first.bit_width

        concat_packed = torch.cat(packed_list, dim=0)
        concat_scales = torch.cat(scales_list, dim=0)
        concat_zp = torch.cat(zp_list, dim=0)

        total_seq_len = sum(t.shape[-2] for t in tensors)
        new_shape = torch.Size([*first.shape[:-2], total_seq_len, first.shape[-1]])
        total_padding = sum(t.padding for t in tensors)

        return CompressedTensor(
            packed_data=concat_packed,
            scales=concat_scales,
            zero_points=concat_zp,
            shape=new_shape,
            bit_width=avg_bw,
            group_size=first.group_size,
            symmetric=first.symmetric,
            dtype=first.dtype,
            padding=total_padding,
            allocations=concat_alloc,
        )


@dataclass
class QuantizationMetrics:
    """Metrics comparing original and dequantized tensors.

    Attributes:
        mse: Mean Squared Error.
        max_abs_error: Maximum absolute element-wise error.
        original_size_bytes: Size of original uncompressed tensor.
        compressed_size_bytes: Size of compressed data structures.
        compression_ratio: Original size / compressed size.
        bits_per_element: Effective bit consumption per element including metadata.
    """

    mse: float
    max_abs_error: float
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    bits_per_element: float


# ── Bit Packing & Unpacking ─────────────────────────────────────────────────

def get_packed_byte_count(numel: int, bit_width: int) -> int:
    """Calculate the byte count produced when packing numel elements at bit_width."""
    if bit_width == 2:
        return (numel + 3) // 4
    elif bit_width == 4:
        return (numel + 1) // 2
    elif bit_width == 3:
        return ((numel + 7) // 8) * 3
    raise InvalidBitWidthError(bit_width)


def pack_bits(quantized: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack integer tensor of values into uint8 byte array.

    Args:
        quantized: 1D uint8 tensor containing quantized integer values in [0, 2^b - 1].
        bit_width: 2, 3, or 4.

    Returns:
        1D uint8 tensor of packed bytes.
    """
    if bit_width not in (2, 3, 4):
        raise InvalidBitWidthError(bit_width)

    numel = quantized.numel()
    if numel == 0:
        return torch.empty(0, dtype=torch.uint8, device=quantized.device)

    quantized = quantized.to(torch.uint8)

    if bit_width == 4:
        if numel % 2 != 0:
            pad = torch.zeros(1, dtype=torch.uint8, device=quantized.device)
            quantized = torch.cat([quantized, pad])
        q = quantized.view(-1, 2)
        return q[:, 0] | (q[:, 1] << 4)

    elif bit_width == 2:
        remainder = numel % 4
        if remainder != 0:
            pad = torch.zeros(4 - remainder, dtype=torch.uint8, device=quantized.device)
            quantized = torch.cat([quantized, pad])
        q = quantized.view(-1, 4)
        return q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)

    else:  # 3-bit
        remainder = numel % 8
        if remainder != 0:
            pad = torch.zeros(8 - remainder, dtype=torch.uint8, device=quantized.device)
            quantized = torch.cat([quantized, pad])
        q = quantized.view(-1, 8)
        v0, v1, v2, v3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        v4, v5, v6, v7 = q[:, 4], q[:, 5], q[:, 6], q[:, 7]

        b0 = v0 | (v1 << 3) | ((v2 & 3) << 6)
        b1 = (v2 >> 2) | (v3 << 1) | (v4 << 4) | ((v5 & 1) << 7)
        b2 = (v5 >> 1) | (v6 << 2) | (v7 << 5)

        return torch.stack([b0, b1, b2], dim=1).reshape(-1)


def unpack_bits(
    packed: torch.Tensor, bit_width: int, target_numel: int
) -> torch.Tensor:
    """Unpack uint8 byte array into integer tensor.

    Args:
        packed: 1D uint8 packed tensor.
        bit_width: 2, 3, or 4.
        target_numel: Expected number of elements after trimming padding.

    Returns:
        1D uint8 tensor of unpacked integer values in [0, 2^b - 1].
    """
    if bit_width not in (2, 3, 4):
        raise InvalidBitWidthError(bit_width)

    if packed.numel() == 0 or target_numel == 0:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)

    if bit_width == 4:
        p = packed.view(-1, 1)
        v0 = p & 0x0F
        v1 = p >> 4
        unpacked = torch.cat([v0, v1], dim=1).reshape(-1)

    elif bit_width == 2:
        p = packed.view(-1, 1)
        v0 = p & 0x03
        v1 = (p >> 2) & 0x03
        v2 = (p >> 4) & 0x03
        v3 = p >> 6
        unpacked = torch.cat([v0, v1, v2, v3], dim=1).reshape(-1)

    else:  # 3-bit
        p = packed.view(-1, 3)
        b0, b1, b2 = p[:, 0], p[:, 1], p[:, 2]

        v0 = b0 & 7
        v1 = (b0 >> 3) & 7
        v2 = (b0 >> 6) | ((b1 & 1) << 2)
        v3 = (b1 >> 1) & 7
        v4 = (b1 >> 4) & 7
        v5 = (b1 >> 7) | ((b2 & 3) << 1)
        v6 = (b2 >> 2) & 7
        v7 = b2 >> 5

        unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).reshape(-1)

    return unpacked[:target_numel].to(torch.uint8)



# ── Base Abstract Quantizer ─────────────────────────────────────────────────

class BaseQuantizer(ABC):
    """Abstract base class for tensor quantizers."""

    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
        bit_width: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        allocations: torch.Tensor | None = None,
    ) -> CompressedTensor:
        """Quantize floating point tensor into compressed representation."""

    @abstractmethod
    def dequantize(self, compressed: CompressedTensor) -> torch.Tensor:
        """Dequantize compressed representation back to floating point tensor."""

    def evaluate(
        self, original: torch.Tensor, compressed: CompressedTensor
    ) -> QuantizationMetrics:
        """Evaluate reconstruction accuracy and compression metrics."""
        dequantized = self.dequantize(compressed)

        orig_f32 = original.to(torch.float32)
        deq_f32 = dequantized.to(torch.float32)

        diff = orig_f32 - deq_f32
        mse = torch.mean(diff ** 2).item()
        max_abs = torch.max(torch.abs(diff)).item()

        return QuantizationMetrics(
            mse=mse,
            max_abs_error=max_abs,
            original_size_bytes=compressed.original_size_bytes,
            compressed_size_bytes=compressed.compressed_size_bytes,
            compression_ratio=compressed.compression_ratio,
            bits_per_element=compressed.bits_per_element,
        )


# ── Group-wise Uniform Quantizer Implementation ─────────────────────────────

class GroupQuantizer(BaseQuantizer):
    """Group-wise uniform quantizer supporting 2-bit, 3-bit, 4-bit, and adaptive mixed-precision.

    Divides tensor into contiguous blocks of `group_size` elements and computes
    individual scale and zero-point parameters for each group.
    """

    def __init__(self, config: QuantizerConfig | None = None) -> None:
        self.config = config or QuantizerConfig()

    def quantize(
        self,
        tensor: torch.Tensor,
        bit_width: int | None = None,
        group_size: int | None = None,
        symmetric: bool | None = None,
        allocations: torch.Tensor | None = None,
    ) -> CompressedTensor:
        """Quantize input tensor into compressed representation.

        Args:
            tensor: Floating-point PyTorch tensor.
            bit_width: Bit precision (overrides config if provided).
            group_size: Group size (overrides config if provided).
            symmetric: Symmetric mode (overrides config if provided).
            allocations: Optional 1D int64 tensor containing assigned bit width per group.

        Returns:
            CompressedTensor object containing packed data and group metadata.
        """
        if tensor.numel() == 0:
            raise EmptyTensorError("Cannot quantize empty tensor.")

        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        device = tensor.device
        gs = group_size if group_size is not None else self.config.group_size
        sym = symmetric if symmetric is not None else self.config.symmetric

        # If per-group allocations are provided, use mixed-precision quantization
        if allocations is not None:
            allocs = allocations.to(device=device, dtype=torch.int64)
            flat_tensor = tensor.reshape(-1)
            num_groups = allocs.numel()

            expected_numel = num_groups * gs
            padding = expected_numel - flat_tensor.numel()
            if padding < 0:
                needed_groups = (flat_tensor.numel() + gs - 1) // gs
                if allocs.numel() < needed_groups:
                    pad_val = allocs[-1].item() if allocs.numel() > 0 else 4
                    allocs = torch.cat([allocs, torch.full((needed_groups - allocs.numel(),), pad_val, dtype=torch.int64, device=device)])
                else:
                    allocs = allocs[:needed_groups]
                num_groups = allocs.numel()
                expected_numel = num_groups * gs
                padding = expected_numel - flat_tensor.numel()

            if padding > 0:
                pad_values = torch.zeros(padding, dtype=orig_dtype, device=device)
                flat_tensor = torch.cat([flat_tensor, pad_values])

            grouped = flat_tensor.view(num_groups, gs).to(torch.float32)
            scales = torch.zeros((num_groups, 1), dtype=orig_dtype, device=device)
            zero_points = torch.zeros((num_groups, 1), dtype=orig_dtype, device=device)

            packed_parts: list[torch.Tensor] = []
            for b in (2, 3, 4):
                idx = torch.nonzero(allocs == b).reshape(-1)
                if idx.numel() == 0:
                    continue
                g_sub = grouped[idx]
                max_val = (1 << b) - 1
                if sym:
                    max_abs = torch.max(torch.abs(g_sub), dim=-1, keepdim=True).values
                    sc = max_abs / (max_val / 2.0)
                    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
                    zp = torch.full_like(sc, max_val / 2.0)
                    q_sub = torch.clamp(torch.round(g_sub / sc + zp), 0, max_val).to(torch.uint8)
                else:
                    min_v = torch.min(g_sub, dim=-1, keepdim=True).values
                    max_v = torch.max(g_sub, dim=-1, keepdim=True).values
                    range_v = max_v - min_v
                    sc = range_v / float(max_val)
                    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
                    zp = torch.round(-min_v / sc)
                    zp = torch.clamp(zp, 0, max_val)
                    q_sub = torch.clamp(torch.round(g_sub / sc + zp), 0, max_val).to(torch.uint8)

                scales[idx] = sc.to(orig_dtype)
                zero_points[idx] = zp.to(orig_dtype)

                packed_b = pack_bits(q_sub.reshape(-1), b)
                packed_parts.append(packed_b)

            packed_data = torch.cat(packed_parts) if len(packed_parts) > 0 else torch.empty(0, dtype=torch.uint8, device=device)
            avg_bw = int(round(allocs.float().mean().item()))

            return CompressedTensor(
                packed_data=packed_data,
                scales=scales,
                zero_points=zero_points,
                shape=orig_shape,
                bit_width=avg_bw,
                group_size=gs,
                symmetric=sym,
                dtype=orig_dtype,
                padding=padding,
                allocations=allocs,
            )

        # Uniform single bit-width quantization path
        bw = bit_width if bit_width is not None else self.config.bit_width
        if bw not in (2, 3, 4):
            raise InvalidBitWidthError(bw)

        numel = tensor.numel()
        pack_align = 4 if bw == 2 else (8 if bw == 3 else 2)
        lcm_align = math.lcm(gs, pack_align)

        remainder = numel % lcm_align
        padding = (lcm_align - remainder) if remainder != 0 else 0

        flat_tensor = tensor.reshape(-1)
        if padding > 0:
            pad_values = torch.zeros(padding, dtype=orig_dtype, device=device)
            flat_tensor = torch.cat([flat_tensor, pad_values])

        grouped = flat_tensor.view(-1, gs).to(torch.float32)
        max_val = (1 << bw) - 1

        if sym:
            max_abs = torch.max(torch.abs(grouped), dim=-1, keepdim=True).values
            scales = max_abs / (max_val / 2.0)
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            zero_points = torch.full_like(scales, max_val / 2.0)
            quantized = torch.clamp(
                torch.round(grouped / scales + zero_points), 0, max_val
            ).to(torch.uint8)
        else:
            min_v = torch.min(grouped, dim=-1, keepdim=True).values
            max_v = torch.max(grouped, dim=-1, keepdim=True).values
            range_v = max_v - min_v
            scales = range_v / float(max_val)
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            zero_points = torch.round(-min_v / scales)
            zero_points = torch.clamp(zero_points, 0, max_val)

            quantized = torch.clamp(
                torch.round(grouped / scales + zero_points), 0, max_val
            ).to(torch.uint8)

        packed_data = pack_bits(quantized.reshape(-1), bw)

        return CompressedTensor(
            packed_data=packed_data,
            scales=scales.to(orig_dtype),
            zero_points=zero_points.to(orig_dtype),
            shape=orig_shape,
            bit_width=bw,
            group_size=gs,
            symmetric=sym,
            dtype=orig_dtype,
            padding=padding,
            allocations=None,
        )

    def dequantize(self, compressed: CompressedTensor) -> torch.Tensor:
        """Dequantize compressed representation back to original float tensor.

        Args:
            compressed: CompressedTensor instance.

        Returns:
            Reconstructed PyTorch float tensor with original shape and dtype.
        """
        gs = compressed.group_size
        padding = compressed.padding

        # Per-group mixed-precision dequantization
        if compressed.allocations is not None:
            allocs = compressed.allocations
            num_groups = allocs.numel()
            device = compressed.packed_data.device
            dtype = compressed.dtype

            scales = compressed.scales.to(torch.float32)
            zero_points = compressed.zero_points.to(torch.float32)

            dequantized_groups = torch.zeros((num_groups, gs), dtype=torch.float32, device=device)

            byte_offset = 0
            for b in (2, 3, 4):
                idx = torch.nonzero(allocs == b).reshape(-1)
                if idx.numel() == 0:
                    continue
                count = idx.numel()
                numel_b = count * gs
                num_bytes_b = get_packed_byte_count(numel_b, b)

                packed_b = compressed.packed_data[byte_offset : byte_offset + num_bytes_b]
                byte_offset += num_bytes_b

                unpacked_b = unpack_bits(packed_b, b, numel_b).view(count, gs).to(torch.float32)
                sc = scales[idx]
                zp = zero_points[idx]
                deq_b = (unpacked_b - zp) * sc
                dequantized_groups[idx] = deq_b

            flat_dequantized = dequantized_groups.reshape(-1)
            if padding > 0:
                flat_dequantized = flat_dequantized[: compressed.original_numel]

            return flat_dequantized.reshape(compressed.shape).to(dtype)

        # Uniform single bit-width dequantization
        bw = compressed.bit_width
        target_numel = compressed.original_numel + padding

        unpacked = unpack_bits(compressed.packed_data, bw, target_numel)
        grouped = unpacked.view(-1, gs).to(torch.float32)
        scales = compressed.scales.to(torch.float32)
        zero_points = compressed.zero_points.to(torch.float32)

        dequantized_grouped = (grouped - zero_points) * scales
        flat_dequantized = dequantized_grouped.reshape(-1)

        if padding > 0:
            flat_dequantized = flat_dequantized[: compressed.original_numel]

        return flat_dequantized.reshape(compressed.shape).to(compressed.dtype)

