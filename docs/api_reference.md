# AdaptiveKV API Reference

This reference documents the primary public classes and functions exported by `adaptivekv`.

---

## Configuration API

```python
from adaptivekv import (
    AdaptiveKVConfig,
    QuantizerConfig,
    AllocationConfig,
    ImportanceConfig,
)
```

### `AdaptiveKVConfig`

Top-level configuration object.

- `quantizer: QuantizerConfig`: quantization engine settings.
- `allocation: AllocationConfig`: bit-allocation strategy settings.
- `importance: ImportanceConfig`: importance scoring settings.
- `device: str`: PyTorch device string (`"cpu"`, `"cuda"`).
- `dtype: str`: working dtype (`"float16"`, `"bfloat16"`, `"float32"`).

### `AllocationConfig`

- `strategy: str`: `"threshold"` or `"budget"`.
- `bits: tuple[int, ...]`: available bit widths, e.g. `(2, 3, 4)`.
- `thresholds: tuple[float, ...]`: quantile cutoffs for `"threshold"` strategy.
- `memory_budget_ratio: float | None`: target memory fraction for `"budget"` strategy (e.g. `0.25`).

---

## Cache API

```python
from adaptivekv import AdaptiveKVCache
```

### `AdaptiveKVCache(config=None, bits=(2, 3, 4), strategy="threshold", memory_budget_ratio=None, group_size=128)`

Hugging Face compatible KV-cache implementation.

- `update(key_states, value_states, layer_idx, cache_kwargs=None)`: updates layer cache and returns dequantized states.
- `get_seq_length(layer_idx=0)`: returns cached sequence length.
- `total_compressed_size_bytes()`: returns total memory consumption in bytes across all layers.
- `overall_compression_ratio()`: returns overall compression ratio relative to FP16.

---

## Quantization Engine API

```python
from adaptivekv import GroupQuantizer, CompressedTensor, QuantizationMetrics
```

### `GroupQuantizer(config=None)`

- `quantize(tensor, bit_width=None, group_size=None, symmetric=None) -> CompressedTensor`: quantizes floating-point tensor.
- `dequantize(compressed: CompressedTensor) -> torch.Tensor`: reconstructs floating-point tensor.
- `evaluate(original, compressed) -> QuantizationMetrics`: computes MSE, max error, compression ratio, and bits/element.

---

## Importance Analyzer API

```python
from adaptivekv import (
    AttentionImportanceAnalyzer,
    MagnitudeImportanceAnalyzer,
    RecencyImportanceAnalyzer,
    create_importance_analyzer,
)
```

- `compute_importance(key_states, value_states, attention_weights=None, group_size=128) -> ImportanceScore`: returns normalized importance scores in `[0.0, 1.0]`.

---

## Model Adapter Integration API

```python
from adaptivekv import apply_adaptive_kv, HuggingFaceAdapter
```

- `apply_adaptive_kv(model, strategy="budget", memory_budget_ratio=0.25)`: validates Hugging Face decoder-only model and binds `AdaptiveKVCache`.
