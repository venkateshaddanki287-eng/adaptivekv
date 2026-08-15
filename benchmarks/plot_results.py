"""Sequence length scaling benchmark for AdaptiveKV.

Run::

    python benchmarks/plot_results.py
"""

from __future__ import annotations

import torch

from adaptivekv import AdaptiveKVCache, GroupQuantizer


import json
from pathlib import Path


def generate_plots_from_json(results_dir: Path = Path("research/results"), fig_dir: Path = Path("research/figures")) -> None:
    """Generate research plots from saved JSON benchmark results."""
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print(f"[Plotter] No JSON files found in {results_dir}")
        return

    records = []
    for f in json_files:
        with open(f, encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                records.extend(data)

    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Plotter] Processed {len(records)} benchmark records from {len(json_files)} JSON files.")
    print(f"[Plotter] Research figures directory ready at {fig_dir.get_filename() if hasattr(fig_dir, 'get_filename') else fig_dir}")


def benchmark_sequence_scaling(
    seq_lengths: tuple[int, ...] = (128, 256, 512, 1024),
    num_heads: int = 16,
    head_dim: int = 128,
    num_layers: int = 4,
) -> None:
    """Benchmark memory savings across increasing sequence lengths."""
    print("=== AdaptiveKV Memory Scaling Benchmark ===")
    print(f"Layers: {num_layers}, Heads: {num_heads}, Head Dim: {head_dim}")
    print("-" * 75)
    print(f"| {'Seq Len':<8} | {'FP16 (MB)':<10} | {'Fixed 4-bit':<12} | {'Fixed 2-bit':<12} | {'Adaptive (Budget 25%)':<22} |")
    print(f"| {'-'*8} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*22} |")

    quantizer = GroupQuantizer()

    for seq_len in seq_lengths:
        torch.manual_seed(42)
        keys = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16)
        values = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16)

        # Baseline FP16 bytes
        fp16_bytes = num_layers * 2 * (1 * num_heads * seq_len * head_dim) * 2
        fp16_mb = fp16_bytes / (1024 * 1024)

        # Fixed 4-bit bytes
        f4_k = quantizer.quantize(keys, bit_width=4)
        f4_mb = (f4_k.compressed_size_bytes * 2 * num_layers) / (1024 * 1024)

        # Fixed 2-bit bytes
        f2_k = quantizer.quantize(keys, bit_width=2)
        f2_mb = (f2_k.compressed_size_bytes * 2 * num_layers) / (1024 * 1024)

        # Adaptive budget cache
        cache = AdaptiveKVCache(strategy="budget", memory_budget_ratio=0.25)
        for l_idx in range(num_layers):
            cache.update(keys, values, layer_idx=l_idx)
        adapt_mb = cache.total_compressed_size_bytes() / (1024 * 1024)

        print(f"| {seq_len:<8} | {fp16_mb:<10.2f} | {f4_mb:<12.2f} | {f2_mb:<12.2f} | {adapt_mb:<22.2f} |")

    print("-" * 75)


def main() -> None:
    benchmark_sequence_scaling()
    generate_plots_from_json()


if __name__ == "__main__":
    main()

