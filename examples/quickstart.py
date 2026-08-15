"""Quickstart example demonstrating AdaptiveKV usage."""

from __future__ import annotations

import torch

from adaptivekv import AdaptiveKVCache, AdaptiveKVConfig, AllocationConfig


def main() -> None:
    print("=== AdaptiveKV Quickstart Example ===")

    # 1. Configure AdaptiveKV with a 25% target memory budget
    config = AdaptiveKVConfig(
        allocation=AllocationConfig(
            strategy="budget",
            memory_budget_ratio=0.25,  # 25% of FP16 memory target
            bits=(2, 3, 4),
        )
    )

    # 2. Instantiate AdaptiveKVCache
    cache = AdaptiveKVCache(config=config)

    # 3. Simulate generation step with 1 layer, 8 heads, 128 sequence length, 64 head dim
    key_states = torch.randn(1, 8, 128, 64, dtype=torch.float16)
    value_states = torch.randn(1, 8, 128, 64, dtype=torch.float16)

    out_keys, out_values = cache.update(key_states, value_states, layer_idx=0)

    print(f"Original KV shape: {key_states.shape}")
    print(f"Compressed cache size: {cache.total_compressed_size_bytes()} bytes")
    print(f"Achieved compression ratio: {cache.overall_compression_ratio():.2f}x")
    print(f"Output key shape for attention: {out_keys.shape}")


if __name__ == "__main__":
    main()
