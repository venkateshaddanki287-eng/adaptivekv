"""Modular command-line interface for AdaptiveKV.

Supports commands:
  - adaptivekv benchmark: Run comparative KV-cache compression benchmarks.
  - adaptivekv compare: Compare specific bit widths or allocation strategies.
  - adaptivekv inspect: Display supported models and system information.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import torch

import adaptivekv
from adaptivekv.integration import SUPPORTED_MODEL_TYPES


def cmd_info(args: argparse.Namespace) -> None:
    """Display library configuration, kernel backend, and package metadata."""
    from adaptivekv.kernels import get_kernel_backend

    print("=== AdaptiveKV Library Information ===")
    print(f"Package Version:       v{adaptivekv.__version__}")
    print(f"PyTorch Version:       {torch.__version__}")
    print(f"Kernel Backend:        {get_kernel_backend()}")
    print(f"CUDA Hardware:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model:             {torch.cuda.get_device_name(0)}")
    print(f"Supported Bit-Widths:  (2, 3, 4)")
    print(f"Allocation Strategies: threshold, budget")
    print(f"Importance Strategies: attention, magnitude, recency")
    print(f"Default Group Size:    128")
    print(f"License:               Apache-2.0")
    print("=" * 38)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run comparative benchmark."""
    import sys
    from pathlib import Path

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from benchmarks.benchmark_kv import run_benchmark
    except ImportError:
        print("[Error] Could not locate 'benchmarks' module. Make sure you run from the repository root directory.")
        return

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


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare quantization schemes for a given tensor shape."""
    from adaptivekv.quantizer import GroupQuantizer

    print("=== AdaptiveKV Quantization Comparison ===")
    print(f"Tensor Shape: (heads={args.heads}, seq_len={args.seq_len}, dim={args.dim})")
    print("-" * 60)

    tensor = torch.randn(args.heads, args.seq_len, args.dim, dtype=torch.float16)
    quantizer = GroupQuantizer()

    for bits in (4, 3, 2):
        compressed = quantizer.quantize(tensor, bit_width=bits, group_size=args.group_size)
        metrics = quantizer.evaluate(tensor, compressed)
        print(
            f"Bit Width: {bits}-bit | "
            f"Ratio: {metrics.compression_ratio:.2f}x | "
            f"MSE: {metrics.mse:.6f} | "
            f"Max Err: {metrics.max_abs_error:.4f} | "
            f"Bits/elem: {metrics.bits_per_element:.2f}"
        )
    print("-" * 60)


def cmd_inspect(args: argparse.Namespace) -> None:
    """Display system info and supported models."""
    print(f"AdaptiveKV v{adaptivekv.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("\nSupported Hugging Face Architectures:")
    for m in SUPPORTED_MODEL_TYPES:
        print(f"  - {m}")


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="adaptivekv",
        description="AdaptiveKV — Dynamic Bit-Allocation for KV-Cache Compression",
    )
    parser.add_argument(
        "--version", action="version", version=f"AdaptiveKV {adaptivekv.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative benchmark")
    p_bench.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    p_bench.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    p_bench.add_argument("--num-heads", type=int, default=16, help="Number of heads")
    p_bench.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    p_bench.add_argument("--runs", type=int, default=3, help="Benchmark runs")
    p_bench.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON results")

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare bit-width precision")
    p_comp.add_argument("--heads", type=int, default=8, help="Attention heads")
    p_comp.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p_comp.add_argument("--dim", type=int, default=128, help="Head dimension")
    p_comp.add_argument("--group-size", type=int, default=128, help="Group size")

    # inspect
    subparsers.add_parser("inspect", help="Display system & model info")

    # info
    subparsers.add_parser("info", help="Display library info and configuration")

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
