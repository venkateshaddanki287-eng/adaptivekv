"""Hugging Face integration example demonstrating AdaptiveKV with transformer models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from adaptivekv.integration import apply_adaptive_kv


class DummyLlamaConfig:
    """Mock Hugging Face model config."""

    model_type = "llama"


class DummyLlamaModel(nn.Module):
    """Mock Hugging Face LlamaModel to test integration without downloading weights."""

    def __init__(self) -> None:
        super().__init__()
        self.config = DummyLlamaConfig()

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        batch, seq_len = input_ids.shape
        num_heads = 4
        head_dim = 32

        # Simulate key and value generation for 1 layer
        keys = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16)
        values = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16)

        if past_key_values is not None:
            _deq_k, _deq_v = past_key_values.update(keys, values, layer_idx=0)
        else:
            _deq_k, _deq_v = keys, values

        logits = torch.randn(batch, seq_len, 1000)
        return logits, past_key_values


def main() -> None:
    print("=== Hugging Face Model Integration Example ===")

    model = DummyLlamaModel()

    # 1. Apply AdaptiveKV to the model with budget strategy (25% target ratio)
    model, kv_cache = apply_adaptive_kv(
        model,
        strategy="budget",
        memory_budget_ratio=0.25,
        bits=(2, 3, 4),
    )

    # 2. Simulate forward pass with token generation
    input_ids = torch.randint(0, 1000, (1, 64))
    logits, kv_cache = model(input_ids, past_key_values=kv_cache)

    print(f"Detected Model Type: {model.config.model_type}")
    print(f"Cached Sequence Length: {kv_cache.get_seq_length(0)}")
    print(f"Compressed Cache Memory: {kv_cache.total_compressed_size_bytes()} bytes")
    print(f"Overall Compression Ratio: {kv_cache.overall_compression_ratio():.2f}x")


if __name__ == "__main__":
    main()
