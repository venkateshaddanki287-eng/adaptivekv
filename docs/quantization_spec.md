# AdaptiveKV Quantization & Bit-Packing Specification

This document defines the mathematical formulation and bit-packing layouts for 2-bit, 3-bit, and 4-bit KV-cache quantization in AdaptiveKV.

---

## 1. Uniform Group-wise Quantization

For a contiguous tensor block $X \in \mathbb{R}^{G}$ of size $G = \text{group\_size}$ and bit-width $b \in \{2, 3, 4\}$, the number of quantization levels is $L = 2^b - 1$:
- 2-bit: $L = 3$, levels $\in \{0, 1, 2, 3\}$
- 3-bit: $L = 7$, levels $\in \{0, 1, 2, 3, 4, 5, 6, 7\}$
- 4-bit: $L = 15$, levels $\in \{0, \dots, 15\}$

### Asymmetric Quantization

$$\text{scale} = \frac{\max(X) - \min(X)}{2^b - 1}$$
$$\text{zero\_point} = \text{clamp}\left(\text{round}\left(-\frac{\min(X)}{\text{scale}}\right), 0, 2^b - 1\right)$$
$$Q = \text{clamp}\left(\text{round}\left(\frac{X}{\text{scale}} + \text{zero\_point}\right), 0, 2^b - 1\right)$$

Dequantization:
$$\hat{X} = (Q - \text{zero\_point}) \times \text{scale}$$

---

## 2. Bit-Packing Layouts

Quantized integer values $Q \in [0, 2^b - 1]$ are stored as `uint8` byte arrays.

### 4-bit Packing Layout (2 values per byte)

Byte $i$ contains two 4-bit values $v_0, v_1$:

$$\text{Byte}_i = (v_0 \ \& \ \text{0x0F}) \ \mid \ ((v_1 \ \& \ \text{0x0F}) \ll 4)$$

Unpacking:
$$v_0 = \text{Byte}_i \ \& \ \text{0x0F}$$
$$v_1 = (\text{Byte}_i \gg 4) \ \& \ \text{0x0F}$$

### 2-bit Packing Layout (4 values per byte)

Byte $i$ contains four 2-bit values $v_0, v_1, v_2, v_3$:

$$\text{Byte}_i = (v_0 \ \& \ 3) \ \mid \ ((v_1 \ \& \ 3) \ll 2) \ \mid \ ((v_2 \ \& \ 3) \ll 4) \ \mid \ ((v_3 \ \& \ 3) \ll 6)$$

Unpacking:
$$v_0 = \text{Byte}_i \ \& \ 3$$
$$v_1 = (\text{Byte}_i \gg 2) \ \& \ 3$$
$$v_2 = (\text{Byte}_i \gg 4) \ \& \ 3$$
$$v_3 = (\text{Byte}_i \gg 6) \ \& \ 3$$

### 3-bit Packing Layout (8 values across 3 bytes)

A block of 8 values $v_0, \dots, v_7 \in [0, 7]$ (24 bits total) packs into 3 bytes $\text{B}_0, \text{B}_1, \text{B}_2$:

$$\text{B}_0 = (v_0 \ \& \ 7) \ \mid \ ((v_1 \ \& \ 7) \ll 3) \ \mid \ ((v_2 \ \& \ 3) \ll 6)$$
$$\text{B}_1 = ((v_2 \gg 2) \ \& \ 1) \ \mid \ ((v_3 \ \& \ 7) \ll 1) \ \mid \ ((v_4 \ \& \ 7) \ll 4) \ \mid \ ((v_5 \ \& \ 1) \ll 7)$$
$$\text{B}_2 = ((v_5 \gg 1) \ \& \ 3) \ \mid \ ((v_6 \ \& \ 7) \ll 2) \ \mid \ ((v_7 \ \& \ 7) \ll 5)$$

Unpacking:
$$v_0 = \text{B}_0 \ \& \ 7$$
$$v_1 = (\text{B}_0 \gg 3) \ \& \ 7$$
$$v_2 = ((\text{B}_0 \gg 6) \ \& \ 3) \ \mid \ ((\text{B}_1 \ \& \ 1) \ll 2)$$
$$v_3 = (\text{B}_1 \gg 1) \ \& \ 7$$
$$v_4 = (\text{B}_1 \gg 4) \ \& \ 7$$
$$v_5 = ((\text{B}_1 \gg 7) \ \& \ 1) \ \mid \ ((\text{B}_2 \ \& \ 3) \ll 1)$$
$$v_6 = (\text{B}_2 \gg 2) \ \& \ 7$$
$$v_7 = (\text{B}_2 \gg 5) \ \& \ 7$$
