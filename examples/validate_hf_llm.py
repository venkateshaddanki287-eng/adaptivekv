"""End-to-end validation experiment comparing standard Hugging Face KV cache vs AdaptiveKV V1.

Measures:
- Actual KV-cache memory usage (bytes and KB)
- Memory reduction percentage (%) and compression ratio
- Number of tokens seen, retained, and evicted
- Generation latency (prefill + decoding time, ms/token)
- Output correctness (Exact Token Match, Jaccard Token Overlap, ROUGE-1 F1, Sequence Length)

Usage::

    python examples/validate_hf_llm.py --model-id facebook/opt-125m --max-new-tokens 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from adaptivekv import (
    AdaptiveKVCache,
    AdaptiveKVConfig,
    AllocationConfig,
    TokenBudgetConfig,
    apply_adaptive_kv,
    compute_cache_statistics,
)


def compute_text_similarity(str1: str, str2: str) -> dict[str, float]:
    """Compute text similarity metrics (Jaccard overlap, ROUGE-1 approximation) between two text strings."""
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())

    if not words1 and not words2:
        jaccard = 1.0
    elif not words1 or not words2:
        jaccard = 0.0
    else:
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / float(union)

    # ROUGE-1 F1 token overlap
    list1 = str1.lower().split()
    list2 = str2.lower().split()
    if not list1 or not list2:
        rouge1_f1 = 0.0
    else:
        match_count = sum(min(list1.count(w), list2.count(w)) for w in set(list1))
        precision = match_count / float(len(list2))
        recall = match_count / float(len(list1))
        if precision + recall > 0:
            rouge1_f1 = (2 * precision * recall) / (precision + recall)
        else:
            rouge1_f1 = 0.0

    return {
        "jaccard_similarity": round(jaccard, 4),
        "rouge1_f1": round(rouge1_f1, 4),
    }


def compute_baseline_kv_bytes(past_key_values: Any) -> int:
    """Calculate total resident bytes in a standard Hugging Face past_key_values cache."""
    total_bytes = 0
    if past_key_values is None:
        return 0

    # DynamicCache in Transformers 5.x (pkv.layers containing DynamicLayer with .keys and .values)
    if hasattr(past_key_values, "layers") and past_key_values.layers:
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                total_bytes += keys.element_size() * keys.nelement() + values.element_size() * values.nelement()
        return total_bytes

    # DynamicCache in Transformers 4.x (pkv.key_cache and pkv.value_cache)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache, strict=False):
            total_bytes += k.element_size() * k.nelement() + v.element_size() * v.nelement()
        return total_bytes

    # Legacy tuple of (key, value) per layer
    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                total_bytes += k.element_size() * k.nelement() + v.element_size() * v.nelement()
        return total_bytes

    return 0


def run_experiment(
    model_id: str = "facebook/opt-125m",
    prompt: str = "Artificial intelligence and machine learning have transformed modern software engineering by enabling",
    max_new_tokens: int = 50,
    device_str: str = "cpu",
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end comparison experiment between standard HF KV cache and AdaptiveKV V1."""
    print("=" * 80)
    print("AdaptiveKV V1 — Hugging Face Causal LLM End-to-End Experiment")
    print("=" * 80)
    print(f"Model ID:        {model_id}")
    print(f"Device:          {device_str}")
    print(f"Max New Tokens:  {max_new_tokens}")
    print("-" * 80)

    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_token_count = int(inputs["input_ids"].shape[1])
    print(f"Prompt ({prompt_token_count} tokens):\n\"{prompt}\"\n")

    # -------------------------------------------------------------------------
    # 1. Baseline Run: Standard Hugging Face KV Cache
    # -------------------------------------------------------------------------
    print(">>> Running Mode 1: Baseline (Standard Hugging Face Cache)...")
    torch.manual_seed(42)
    t0 = time.perf_counter()
    with torch.no_grad():
        base_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    t1 = time.perf_counter()
    base_latency = t1 - t0
    base_generated_tokens = base_out.sequences[0]
    base_text = tokenizer.decode(base_generated_tokens, skip_special_tokens=True)
    base_kv_bytes = compute_baseline_kv_bytes(base_out.past_key_values)
    base_tokens_generated = len(base_generated_tokens) - prompt_token_count
    base_ms_per_token = (base_latency * 1000.0) / max(1, base_tokens_generated)

    print(f"  Latency:           {base_latency:.4f} s ({base_ms_per_token:.2f} ms/token)")
    print(f"  KV Cache Memory:   {base_kv_bytes / 1024.0:.2f} KB ({base_kv_bytes:,} bytes)")
    print(f"  Tokens Generated:  {base_tokens_generated}")
    print(f"  Output Text:\n    \"{base_text}\"\n")

    # -------------------------------------------------------------------------
    # Define AdaptiveKV Test Configurations
    # -------------------------------------------------------------------------
    test_configs = [
        {
            "name": "AdaptiveKV (Quantization Only)",
            "kwargs": {
                "enable_quantization": True,
                "enable_adaptive_bits": True,
                "enable_token_eviction": False,
            },
        },
        {
            "name": "AdaptiveKV (Eviction Only: Keep 50%)",
            "kwargs": {
                "enable_quantization": False,
                "enable_adaptive_bits": False,
                "enable_token_eviction": True,
                "max_cache_tokens": 48,
                "keep_ratio": 0.5,
                "recent_window": 16,
                "sink_tokens": 4,
            },
        },
        {
            "name": "AdaptiveKV V1 Full (Eviction 50% + Quantization 2/3/4-bit)",
            "kwargs": {
                "enable_quantization": True,
                "enable_adaptive_bits": True,
                "enable_token_eviction": True,
                "max_cache_tokens": 48,
                "keep_ratio": 0.5,
                "recent_window": 16,
                "sink_tokens": 4,
            },
        },
    ]

    experiment_results: list[dict[str, Any]] = [
        {
            "mode": "Baseline (Standard Cache)",
            "latency_s": round(base_latency, 4),
            "ms_per_token": round(base_ms_per_token, 2),
            "kv_memory_bytes": base_kv_bytes,
            "kv_memory_kb": round(base_kv_bytes / 1024.0, 2),
            "memory_saved_percent": 0.0,
            "compression_ratio": 1.0,
            "tokens_seen": (prompt_token_count + base_tokens_generated) * getattr(model.config, "num_hidden_layers", 12),
            "tokens_retained": (prompt_token_count + base_tokens_generated) * getattr(model.config, "num_hidden_layers", 12),
            "tokens_evicted": 0,
            "token_retention_ratio": 1.0,
            "exact_token_match_percent": 100.0,
            "jaccard_similarity": 1.0,
            "rouge1_f1": 1.0,
            "generated_text": base_text,
        }
    ]

    # -------------------------------------------------------------------------
    # 2. AdaptiveKV Configurations Runs
    # -------------------------------------------------------------------------
    for idx, cfg_info in enumerate(test_configs, start=2):
        mode_name = cfg_info["name"]
        kwargs = cfg_info["kwargs"]
        print(f">>> Running Mode {idx}: {mode_name}...")

        # Re-instantiate clean model adapter and fresh cache
        fresh_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        fresh_model.eval()
        adapted_model, cache = apply_adaptive_kv(fresh_model, **kwargs)

        torch.manual_seed(42)
        t0 = time.perf_counter()
        with torch.no_grad():
            akv_out = adapted_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=cache,
                return_dict_in_generate=True,
            )
        t1 = time.perf_counter()
        akv_latency = t1 - t0

        akv_generated_tokens = akv_out.sequences[0]
        akv_text = tokenizer.decode(akv_generated_tokens, skip_special_tokens=True)
        akv_tokens_generated = len(akv_generated_tokens) - prompt_token_count
        akv_ms_per_token = (akv_latency * 1000.0) / max(1, akv_tokens_generated)

        # Retrieve cache performance metrics
        stats = compute_cache_statistics(cache)
        akv_kv_bytes = cache.total_compressed_size_bytes()
        mem_saved_pct = ((base_kv_bytes - akv_kv_bytes) / float(base_kv_bytes)) * 100.0 if base_kv_bytes > 0 else 0.0
        comp_ratio = base_kv_bytes / float(max(1, akv_kv_bytes))

        # Compute token match & similarity metrics
        min_len = min(len(base_generated_tokens), len(akv_generated_tokens))
        matches = (base_generated_tokens[:min_len] == akv_generated_tokens[:min_len]).sum().item()
        exact_match_pct = (matches / float(max(1, min_len))) * 100.0
        sim_metrics = compute_text_similarity(base_text, akv_text)

        print(f"  Latency:           {akv_latency:.4f} s ({akv_ms_per_token:.2f} ms/token)")
        print(f"  KV Cache Memory:   {akv_kv_bytes / 1024.0:.2f} KB ({akv_kv_bytes:,} bytes)")
        print(f"  Memory Saved:      {mem_saved_pct:.2f}% (Compression: {comp_ratio:.2f}x)")
        print(f"  Tokens Retained:   {stats['tokens_currently_cached']} / {stats['tokens_seen']} (Evicted: {stats['tokens_evicted']})")
        print(f"  Exact Token Match: {matches}/{min_len} ({exact_match_pct:.2f}%)")
        print(f"  Jaccard Sim / ROUGE-1 F1: {sim_metrics['jaccard_similarity']} / {sim_metrics['rouge1_f1']}")
        print(f"  Output Text:\n    \"{akv_text}\"\n")

        record = {
            "mode": mode_name,
            "latency_s": round(akv_latency, 4),
            "ms_per_token": round(akv_ms_per_token, 2),
            "kv_memory_bytes": akv_kv_bytes,
            "kv_memory_kb": round(akv_kv_bytes / 1024.0, 2),
            "memory_saved_percent": round(mem_saved_pct, 2),
            "compression_ratio": round(comp_ratio, 2),
            "tokens_seen": stats["tokens_seen"],
            "tokens_retained": stats["tokens_currently_cached"],
            "tokens_evicted": stats["tokens_evicted"],
            "token_retention_ratio": round(stats["token_retention_ratio"], 4),
            "exact_token_match_percent": round(exact_match_pct, 2),
            "jaccard_similarity": sim_metrics["jaccard_similarity"],
            "rouge1_f1": sim_metrics["rouge1_f1"],
            "generated_text": akv_text,
        }
        experiment_results.append(record)

    # -------------------------------------------------------------------------
    # Print Markdown Comparative Summary Table
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("EXPERIMENTAL SUMMARY TABLE")
    print("=" * 100)
    headers = [
        "Mode",
        "Memory (KB)",
        "Saved (%)",
        "Ratio",
        "Tokens (Ret/Evic)",
        "Latency (s)",
        "Token Match (%)",
        "ROUGE-1 F1",
    ]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---'] * len(headers))} |")
    for r in experiment_results:
        ret_evic = f"{r['tokens_retained']}/{r['tokens_evicted']}"
        row = [
            r["mode"],
            f"{r['kv_memory_kb']:.1f}",
            f"{r['memory_saved_percent']:.1f}%",
            f"{r['compression_ratio']:.2f}x",
            ret_evic,
            f"{r['latency_s']:.2f}s",
            f"{r['exact_token_match_percent']:.1f}%",
            f"{r['rouge1_f1']:.4f}",
        ]
        print(f"| {' | '.join(row)} |")
    print("=" * 100)

    summary_payload = {
        "model_id": model_id,
        "prompt": prompt,
        "prompt_tokens": prompt_token_count,
        "max_new_tokens": max_new_tokens,
        "device": device_str,
        "results": experiment_results,
    }

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\n[Experiment] Full results saved to: {out_path.resolve()}")

    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AdaptiveKV Hugging Face LLM Validation Experiment")
    parser.add_argument("--model-id", type=str, default="facebook/opt-125m", help="Hugging Face model ID")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial intelligence and machine learning have transformed modern software engineering by enabling",
        help="Input text prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--output-file", type=str, default="research/results/hf_validation_experiment.json", help="Path for JSON output")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(
        model_id=args.model_id,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device_str=device_str,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
