"""Benchmarking script for comparing FP16 baseline, fixed 2/3/4-bit, and AdaptiveKV schemes.

Run::

    python benchmarks/benchmark_kv.py --seq-len 512 --num-layers 4
"""

from __future__ import annotations

import argparse
import time

import torch

from adaptivekv import (
    AdaptiveKVCache,
    AdaptiveKVConfig,
    AllocationConfig,
    GroupQuantizer,
    compute_quality_metrics,
)


import json
from pathlib import Path


def run_benchmark(
    seq_len: int = 512,
    num_heads: int = 16,
    head_dim: int = 128,
    num_layers: int = 4,
    batch_size: int = 1,
    num_runs: int = 5,
    device_str: str = "cpu",
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Run comparative benchmark across FP16 baseline, fixed bit, and AdaptiveKV caches."""
    print("=== AdaptiveKV Comparative Benchmark ===")
    print(f"Config: layers={num_layers}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}, device={device_str}")
    print("-" * 80)

    device = torch.device(device_str)
    dtype = torch.float16

    # Generate synthetic KV cache states
    torch.manual_seed(42)
    keys_list = [
        torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        for _ in range(num_layers)
    ]
    values_list = [
        torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        for _ in range(num_layers)
    ]

    # Baseline FP16 size
    total_elements = num_layers * 2 * (batch_size * num_heads * seq_len * head_dim)
    fp16_bytes = total_elements * 2  # 2 bytes per float16

    quantizer = GroupQuantizer()

    configs = [
        ("FP16 Baseline", None),
        ("Fixed 4-bit", 4),
        ("Fixed 3-bit", 3),
        ("Fixed 2-bit", 2),
        ("AdaptiveKV (Threshold)", "threshold"),
        ("AdaptiveKV (Budget 25%)", "budget"),
    ]

    raw_results: list[dict] = []
    display_rows: list[dict[str, str]] = []

    for name, mode in configs:
        mse_sum = 0.0
        max_abs_sum = 0.0
        cos_sim_sum = 0.0
        compressed_bytes = 0
        latencies = []
        bit_counts = {2: 0, 3: 0, 4: 0}

        for _run in range(num_runs):
            start_t = time.perf_counter()

            if mode is None:  # FP16 Baseline
                compressed_bytes = fp16_bytes
                mse_sum += 0.0
                max_abs_sum += 0.0
                cos_sim_sum += 1.0
                bit_counts = {16: total_elements}
            elif isinstance(mode, int):  # Fixed Bit
                layer_bytes = 0
                for k, v in zip(keys_list, values_list, strict=True):
                    ck = quantizer.quantize(k, bit_width=mode)
                    cv = quantizer.quantize(v, bit_width=mode)
                    layer_bytes += ck.compressed_size_bytes + cv.compressed_size_bytes

                    deq_k = quantizer.dequantize(ck)
                    m = compute_quality_metrics(k, deq_k)
                    mse_sum += m.mse / num_layers
                    max_abs_sum += m.max_abs_error / num_layers
                    cos_sim_sum += m.cosine_similarity / num_layers
                compressed_bytes = layer_bytes
                bit_counts = {mode: total_elements}
            else:  # AdaptiveKV
                if mode == "threshold":
                    cfg = AdaptiveKVConfig(
                        allocation=AllocationConfig(strategy="threshold")
                    )
                else:
                    cfg = AdaptiveKVConfig(
                        allocation=AllocationConfig(strategy="budget", memory_budget_ratio=0.25)
                    )
                cache = AdaptiveKVCache(config=cfg)
                for l_idx, (k, v) in enumerate(zip(keys_list, values_list, strict=True)):
                    deq_k, _ = cache.update(k, v, layer_idx=l_idx)
                    m = compute_quality_metrics(k, deq_k)
                    mse_sum += m.mse / num_layers
                    max_abs_sum += m.max_abs_error / num_layers
                    cos_sim_sum += m.cosine_similarity / num_layers

                    layer_cache = cache.layers[l_idx]
                    if layer_cache.last_allocation is not None:
                        allocs = layer_cache.last_allocation.allocations
                        for b in (2, 3, 4):
                            bit_counts[b] = bit_counts.get(b, 0) + int((allocs == b).sum().item())

                compressed_bytes = cache.total_compressed_size_bytes()

            end_t = time.perf_counter()
            latencies.append(end_t - start_t)

        avg_latency_ms = (sum(latencies) / num_runs) * 1000.0
        tps = (seq_len / (sum(latencies) / num_runs)) if sum(latencies) > 0 else 0.0
        avg_mse = mse_sum / num_runs
        avg_max_abs = max_abs_sum / num_runs
        avg_cos_sim = cos_sim_sum / num_runs
        comp_ratio = fp16_bytes / max(1, compressed_bytes)
        mem_saved_pct = (1.0 - compressed_bytes / fp16_bytes) * 100.0

        record = {
            "model": "synthetic-transformer",
            "method": name,
            "context_length": seq_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "original_bytes": fp16_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(comp_ratio, 4),
            "memory_saved_percent": round(mem_saved_pct, 4),
            "latency_ms": round(avg_latency_ms, 4),
            "tokens_per_second": round(tps, 2),
            "mse": round(avg_mse, 8),
            "max_abs_error": round(avg_max_abs, 6),
            "cosine_similarity": round(avg_cos_sim, 6),
            "bit_distribution": bit_counts,
        }
        raw_results.append(record)

        display_rows.append({
            "Method": name,
            "Memory (KB)": f"{compressed_bytes / 1024.0:.1f}",
            "Ratio": f"{comp_ratio:.2f}x",
            "Saved": f"{mem_saved_pct:.1f}%",
            "MSE": f"{avg_mse:.6f}",
            "Cos Sim": f"{avg_cos_sim:.4f}",
            "Latency (ms)": f"{avg_latency_ms:.2f}",
        })

    # Print markdown results table
    headers = ["Method", "Memory (KB)", "Ratio", "Saved", "MSE", "Cos Sim", "Latency (ms)"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---'] * len(headers))} |")
    for row in display_rows:
        vals = [row[h] for h in headers]
        print(f"| {' | '.join(vals)} |")
    print("-" * 80)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_file = out_path / f"benchmark_ctx_{seq_len}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2)
        print(f"[Benchmark] Saved JSON results to {json_file}")

    return raw_results


def main() -> None:
    parser = argparse.ArgumentParser(description="AdaptiveKV Comparative Benchmark")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--output-dir", type=str, default="research/results", help="Output directory for JSON results")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    run_benchmark(
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_runs=args.runs,
        device_str=device_str,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

