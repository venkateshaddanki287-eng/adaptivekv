# AdaptiveKV Architecture Specification

AdaptiveKV is designed as a modular, low-overhead layer for dynamic Key-Value (KV) cache memory compression during LLM inference.

```
                  ┌─────────────────────────────────────────┐
                  │          Hugging Face LLM               │
                  └────────────────────┬────────────────────┘
                                       │ (past_key_values)
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │            AdaptiveKVCache              │
                  └────────────────────┬────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │   LayerKVCache (Layer 0)  │   . . . . . .   │   LayerKVCache (Layer N)  │
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
      ┌─────────┴─────────┐                         ┌─────────┴─────────┐
      ▼                   ▼                         ▼                   ▼
┌───────────┐       ┌───────────┐             ┌───────────┐       ┌───────────┐
│ Key Cache │       │ Value Cache│            │ Key Cache │       │ Value Cache│
└─────┬─────┘       └─────┬─────┘             └─────┬─────┘       └─────┬─────┘
      │                   │                         │                   │
      └─────────┬─────────┘                         └─────────┬─────────┘
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │  1. Importance Analyzer   │                 │  1. Importance Analyzer   │
  │     (Attention/Magnitude) │                 │     (Attention/Magnitude) │
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │  2. Adaptive Allocator    │                 │  2. Adaptive Allocator    │
  │     (Threshold/Budget)    │                 │     (Threshold/Budget)    │
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │  3. GroupQuantizer        │                 │  3. GroupQuantizer        │
  │     (2, 3, 4-bit Packing) │                 │     (2, 3, 4-bit Packing) │
  └───────────────────────────┘                 └───────────────────────────┘
```

---

## Core Components

### 1. Configuration System (`config.py`)

All settings are encapsulated in frozen, validated dataclasses:
- `QuantizerConfig`: controls `bit_width`, `group_size`, and `symmetric` quantization.
- `AllocationConfig`: controls allocation `strategy` (`threshold` vs `budget`), available bit levels `(2, 3, 4)`, decision thresholds, and `memory_budget_ratio`.
- `ImportanceConfig`: controls importance scoring strategy (`attention`, `magnitude`, `recency`) and score normalization.
- `AdaptiveKVConfig`: top-level container uniting all sub-configurations.

### 2. Quantization Engine (`quantizer.py`)

The quantizer operates on tensor blocks of size `group_size` (default: 128 elements):
- Computes group-wise floating-point scale factors and integer zero-points.
- Packs quantized integer values into contiguous uint8 byte arrays:
  - **4-bit**: 2 values per byte.
  - **3-bit**: 8 values packed across 3 bytes (24 bits total).
  - **2-bit**: 4 values per byte.
- Handles padding alignment automatically for arbitrary tensor shapes.

### 3. Importance Analyzer (`importance.py`)

Computes group-wise importance scores $S_g \in [0.0, 1.0]$:
- **Attention Strategy**: Accumulates attention weights $A_{b, h, q, k}$ over query tokens and attention heads. If attention weights are unavailable, falls back to key vector L2 norms $\|K_k\|_2$.
- **Magnitude Strategy**: Sums $\|K_k\|_2 + \|V_k\|_2$ vector norms.
- **Recency Strategy**: Applies temporal recency weighting $k / N$.

### 4. Adaptive Bit Allocator (`allocator.py`)

Maps group importance scores $S_g$ to bit widths $b_g \in \{2, 3, 4\}$:
- **Threshold Strategy**: Categorizes scores using user-defined quantile boundaries.
- **Budget Strategy**: Solves a greedy marginal-gain resource allocation optimization problem ($O(N \log N)$) to satisfy a target memory budget ratio $\gamma \cdot 16.0$ bits/element while maximizing total output quality.

### 5. Adaptive Cache Interface (`cache.py`)

Subclasses Hugging Face `Cache` to provide plug-and-play integration with decoder-only models. Maintains compressed layer caches, handles autoregressive decoding updates, and dequantizes states on the fly during attention computation.
