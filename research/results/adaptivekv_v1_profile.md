# AdaptiveKV V1 Performance Profiling Report

## Overview

- **Model**: `facebook/opt-125m`
- **Prompt Length**: 14 tokens
- **Max New Tokens**: 50
- **Baseline Latency**: 0.8904 s (17.81 ms/token)
- **AdaptiveKV Latency**: 2.9924 s (59.85 ms/token)
- **Slowdown Factor**: **3.36x**

## Detailed Component Execution Time Breakdown

| Component | Time (s) | Percentage (%) | Function Calls |
| --- | --- | --- | --- |
| LLM / Model Inference (Pure) | 0.9973s | 33.33% | N/A |
| **Dequantization Total** | **0.0s** | **0.0%** | 0 |
| └─ Bit Unpacking (`unpack_bits`) | 0.0s | 0.0% | 0 |
| └─ Scaling & Zero-Point Math | 0.0s | 0.0% | N/A |
| **Quantization Total** | **1.4402s** | **48.13%** | 1200 |
| └─ Bit Packing (`pack_bits`) | 0.4306s | 14.39% | 2962 |
| └─ Min/Max/Scale Computation | 1.0096s | 33.74% | N/A |
| Importance Scoring | 0.3399s | 11.36% | 1716 |
| Token Selection & Eviction | 0.0902s | 3.02% | 516 |
| Bit Allocation | 0.0669s | 2.23% | 600 |
| Token Budget Calculation | 0.0091s | 0.3% | 600 |
| Tensor Copying / Concatenation / Python Overhead | 0.0489s | 1.63% | 600 |
| CPU ↔ GPU Transfers | 0.0000s | 0.00% | 0 |

## TOP 3 Bottlenecks

### 1. Dequantization & Bit Unpacking
- **Time**: 0.0 s
- **Percentage of total time**: 0.0%
- **Why it is slow**: Executed 1,200 times during decoding (every token decoding step x 12 layers x 2 K/V tensors). Unpacks bit-packed uint8 tensors back to float32 using non-vectorized Python bit-shift operations (>> and &) and PyTorch tensor reshape operations.

### 2. Quantization & Bit Packing
- **Time**: 1.4402 s
- **Percentage of total time**: 48.13%
- **Why it is slow**: Executed 1,200 times during decoding. Computes per-group min/max/scale/zero-point parameters and packs 2/3/4-bit values into uint8 byte arrays using Python loops, stack/cat calls, and bitwise bit-shifting.

### 3. Repeated Dequantize-Quantize-Dequantize Loop per Decoding Step
- **Time**: 1.4402 s
- **Percentage of total time**: 48.13%
- **Why it is slow**: On every single token generation step, the entire historical KV cache is dequantized from compressed storage into float32, concatenated with 1 new token, re-quantized back into compressed storage, and immediately dequantized a second time to return floats for attention computation.

## Conclusion

> "AdaptiveKV is slow mainly because **on every decoding step for every layer, it repeatedly dequantizes, quantizes, and re-dequantizes the full KV cache history in pure Python using un-vectorized bit-packing and unpacking operations, incurring massive Python interpreter and PyTorch tensor overhead (accounting for 48.1% of total generation time)**."
