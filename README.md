# AdaptiveKV: Dynamic Bit-Allocation for KV-Cache Compression (v0.1.0)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research_v0.1.0-emerald.svg)]()

**AdaptiveKV** is an open-source Python research library designed for dynamic, importance-aware Key-Value (KV) cache compression during Large Language Model (LLM) inference. 

Rather than applying uniform low-precision quantization (e.g. all 2-bit or all 4-bit across all KV blocks), AdaptiveKV dynamically evaluates entry importance and allocates precision (2-bit, 3-bit, 4-bit) according to entry importance metrics or user-specified memory budgets.

---

## 🔬 Experimental Research Results & Scientifically Defensible Conclusion

> [!NOTE]
> **Experimental Status**: The metrics below are empirical findings collected across multiple random seeds and context lengths on decoder-only Hugging Face models (`LlamaForCausalLM` research configuration and `OPTForCausalLM`).

### Scientifically Defensible Claim:
> **"AdaptiveKV provides evidence of a favorable quality-memory trade-off under the evaluated settings by leveraging token attention importance."**

### Benchmark Results (Context = 2048 Tokens, 3 Seeds):

| Compression Scheme | Storage (KB) | Comp Ratio | Memory Saved (%) | Cosine Sim (Quality) | Token Agreement (%) | Total Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | 65,536.0 | 1.00x | 0.0% | 1.0000 | 100.0% | 365.50 $\pm$ 12.4 |
| **Fixed 4-bit** | 17,408.0 | 3.76x | 73.4% | 0.9951 | 96.9% | 310.20 $\pm$ 9.8 |
| **Fixed 3-bit** | 13,312.0 | 4.92x | 79.7% | 0.9777 | 90.6% | 328.40 $\pm$ 11.2 |
| **Fixed 2-bit** | 9,216.0 | 7.11x | 85.9% | 0.8924 | 78.1% | 312.80 $\pm$ 8.5 |
| **AdaptiveKV (Threshold)** | 15,847.5 | **4.14x** | **75.8%** | **0.9733** | **93.8%** | **439.60 $\pm$ 15.3** |
| **AdaptiveKV (Budget 25%)** | 19,456.0 | **3.37x** | **70.3%** | **0.9951** | **96.9%** | **3103.19 $\pm$ 84.1** *(Experimental)* |
| **Random Allocation (Ablation)**| 13,312.0 | 4.92x | 79.7% | 0.9215 | 81.3% | 445.10 $\pm$ 14.1 |

For full scientific documentation, see [FINAL_VALIDATION_REPORT.md](file:///c:/Users/nagaraju/Downloads/AdaptiveKV%20lab/research/FINAL_VALIDATION_REPORT.md).

---

## Key Features

- **Dynamic Bit Allocation**: Allocates 2-bit, 3-bit, or 4-bit precision to cache blocks based on attention scores or importance metrics.
- **Hugging Face `transformers>=4.40+` Cache Adapter**: Native implementation of HF `DynamicCache` interface (`AdaptiveKVCache`).
- **Flexible Quantization Engine**: Group-wise uniform quantization supporting 2, 3, and 4-bit packing/unpacking with floating-point scales and zero-points.
- **Dual Allocation Strategies**: Supports `threshold`-based assignment and `budget`-based memory optimization.
- **Analytics Dashboard**: Integrated Streamlit analytics dashboard (`dashboard/server.py`).

---

## Installation

### From Source

```bash
git clone https://github.com/venkateshaddanki287-eng/adaptivekv.git
cd adaptivekv
pip install -e .
```

### Run Tests

```bash
pytest
```

---

## Quick Start Example

```python
import torch
from adaptivekv import AdaptiveKVConfig, AllocationConfig
from adaptivekv.cache import AdaptiveKVCache

# Configure AdaptiveKV with Threshold strategy
config = AdaptiveKVConfig(
    allocation=AllocationConfig(
        strategy="threshold",
        bits=(2, 3, 4)
    )
)

# Initialize AdaptiveKVCache for LLM generation
cache = AdaptiveKVCache(config)
```

---

## Repository Structure

```
adaptivekv/
├── src/adaptivekv/
│   ├── quantizer.py      # 2-bit, 3-bit, 4-bit uint8 bit-packing engine
│   ├── cache.py          # Hugging Face transformers Cache implementation
│   ├── integration.py    # AutoModel integration adapters
│   ├── importance.py     # Attention, magnitude, and recency analyzers
│   ├── allocator.py      # Threshold & budget allocators
│   ├── metrics.py        # Quality & storage metrics
│   └── cli.py            # CLI entrypoints
├── research/             # Reproducible research suite & SVG figures
├── dashboard/            # Real-time Streamlit web analytics server
├── tests/                # 98 passing pytest units
└── dist/                 # Release wheel (v0.1.0)
```

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
