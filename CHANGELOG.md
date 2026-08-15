# Changelog

All notable changes to the `adaptivekv` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-15

### Added
- **Mixed-Precision Per-Group Quantization**:
  - Custom uint8 bit-packing for 2-bit, 3-bit, and 4-bit precision.
  - Per-group symmetric quantization scaling and zero-point alignment (`GroupQuantizer`).
  - Reconstruction metric utilities computing MSE, MAE, SNR, and Cosine Similarity (`compute_quality_metrics`).
- **Token Importance Analyzers**:
  - `AttentionImportanceAnalyzer`: Dynamically tracks attention weights from transformer decoder layers.
  - `MagnitudeImportanceAnalyzer`: Fallback analyzer based on key-value tensor norm magnitudes.
  - `RecencyImportanceAnalyzer`: Positional recency decay score for streaming inputs.
- **Adaptive Bit Allocators**:
  - `ThresholdAllocator`: Vectorized score bucketing into 2-bit, 3-bit, and 4-bit precision levels.
  - `BudgetAllocator`: Greedy marginal quality gain optimization under target memory budget constraints.
- **Hugging Face `transformers>=4.40+` Cache Integration**:
  - `AdaptiveKVCache` implementing full HF `DynamicCache` interface (`get_mask_sizes`, `is_compileable`, `__len__`, `__getitem__`).
  - Helper functions `apply_adaptive_kv` and `is_model_supported` for standard decoder models (`LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM`, `GemmaForCausalLM`, `OPTForCausalLM`).
- **Command Line Interface (CLI)**:
  - `adaptivekv info`, `adaptivekv inspect`, `adaptivekv compare`, `adaptivekv benchmark`.
- **Interactive Analytics Dashboard**:
  - Real-time Streamlit analytics server (`dashboard/server.py`).
- **Empirical Research Experiment Suite**:
  - Scientific experiment runners (`research/experiments/run_research_experiment.py`, `run_research_experiment_v2.py`).
  - SVG vector figure generators and CSV/Markdown report generators.
  - Full research validation report (`research/FINAL_VALIDATION_REPORT.md`).

### Scientific Status & Research Conclusion
> *"AdaptiveKV provides evidence of a favorable quality-memory trade-off under the evaluated settings by leveraging token attention importance."*
