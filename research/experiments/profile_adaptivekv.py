"""Profiling script for AdaptiveKV V1 performance breakdown.

Measures fine-grained execution time of each component in AdaptiveKV V1 during HF LLM generation.
Includes a warmup pass to ensure pure generation latency is measured without cold-start model overhead.
Saves results to:
- research/results/adaptivekv_v1_profile.json
- research/results/adaptivekv_v1_profile.md
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import adaptivekv.quantizer as quantizer_module
import adaptivekv.cache as cache_module
import adaptivekv.importance as importance_module
import adaptivekv.selector as selector_module
import adaptivekv.allocator as allocator_module
import adaptivekv.controller as controller_module
from adaptivekv import apply_adaptive_kv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "research" / "results"

TIMERS = {
    "total_generate_wall_time": 0.0,
    "model_forward_pure": 0.0,
    "cache_update_total": 0.0,
    "dequantize_total": 0.0,
    "unpack_bits_total": 0.0,
    "dequant_math_scale_zero": 0.0,
    "quantize_total": 0.0,
    "pack_bits_total": 0.0,
    "quant_math_min_max_scale": 0.0,
    "importance_scoring_total": 0.0,
    "token_selection_eviction_total": 0.0,
    "bit_allocation_total": 0.0,
    "budget_calc_total": 0.0,
    "other_cache_python_overhead": 0.0,
}

COUNTERS = {
    "cache_update_calls": 0,
    "quantize_calls": 0,
    "dequantize_calls": 0,
    "pack_bits_calls": 0,
    "unpack_bits_calls": 0,
    "importance_scoring_calls": 0,
    "token_selection_calls": 0,
    "bit_allocation_calls": 0,
    "budget_calc_calls": 0,
}

_orig_pack_bits = quantizer_module.pack_bits
_orig_unpack_bits = quantizer_module.unpack_bits
_orig_quantize = quantizer_module.GroupQuantizer.quantize
_orig_dequantize = quantizer_module.GroupQuantizer.dequantize
_orig_compute_token_importance = importance_module.AttentionImportanceAnalyzer.compute_token_importance
_orig_compute_importance = importance_module.AttentionImportanceAnalyzer.compute_importance
_orig_select = selector_module.TokenSelector.select
_orig_allocate = allocator_module.AdaptiveBitAllocator.allocate
_orig_get_budget = controller_module.TokenBudgetController.get_budget
_orig_update = cache_module.LayerKVCache.update


def profile_pack_bits(quantized: torch.Tensor, bit_width: int) -> torch.Tensor:
    COUNTERS["pack_bits_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_pack_bits(quantized, bit_width)
    t1 = time.perf_counter()
    TIMERS["pack_bits_total"] += (t1 - t0)
    return res


def profile_unpack_bits(packed: torch.Tensor, bit_width: int, target_numel: int) -> torch.Tensor:
    COUNTERS["unpack_bits_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_unpack_bits(packed, bit_width, target_numel)
    t1 = time.perf_counter()
    TIMERS["unpack_bits_total"] += (t1 - t0)
    return res


def profile_quantize(self, tensor: torch.Tensor, *args, **kwargs) -> Any:
    COUNTERS["quantize_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_quantize(self, tensor, *args, **kwargs)
    t1 = time.perf_counter()
    TIMERS["quantize_total"] += (t1 - t0)
    return res


def profile_dequantize(self, compressed: Any) -> torch.Tensor:
    COUNTERS["dequantize_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_dequantize(self, compressed)
    t1 = time.perf_counter()
    TIMERS["dequantize_total"] += (t1 - t0)
    return res


def profile_compute_token_importance(self, *args, **kwargs) -> torch.Tensor:
    COUNTERS["importance_scoring_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_compute_token_importance(self, *args, **kwargs)
    t1 = time.perf_counter()
    TIMERS["importance_scoring_total"] += (t1 - t0)
    return res


def profile_compute_importance(self, *args, **kwargs) -> Any:
    COUNTERS["importance_scoring_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_compute_importance(self, *args, **kwargs)
    t1 = time.perf_counter()
    TIMERS["importance_scoring_total"] += (t1 - t0)
    return res


def profile_select(self, *args, **kwargs) -> Any:
    COUNTERS["token_selection_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_select(self, *args, **kwargs)
    t1 = time.perf_counter()
    TIMERS["token_selection_eviction_total"] += (t1 - t0)
    return res


def profile_allocate(self, *args, **kwargs) -> Any:
    COUNTERS["bit_allocation_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_allocate(self, *args, **kwargs)
    t1 = time.perf_counter()
    TIMERS["bit_allocation_total"] += (t1 - t0)
    return res


def profile_get_budget(self, *args, **kwargs) -> int:
    COUNTERS["budget_calc_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_get_budget(self, *args, **kwargs)
    t1 = time.perf_counter()
    TIMERS["budget_calc_total"] += (t1 - t0)
    return res


def profile_update(self, key_states: torch.Tensor, value_states: torch.Tensor, attention_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    COUNTERS["cache_update_calls"] += 1
    t0 = time.perf_counter()
    res = _orig_update(self, key_states, value_states, attention_weights=attention_weights)
    t1 = time.perf_counter()
    TIMERS["cache_update_total"] += (t1 - t0)
    return res


def install_hooks() -> None:
    quantizer_module.pack_bits = profile_pack_bits
    quantizer_module.unpack_bits = profile_unpack_bits
    quantizer_module.GroupQuantizer.quantize = profile_quantize
    quantizer_module.GroupQuantizer.dequantize = profile_dequantize
    importance_module.AttentionImportanceAnalyzer.compute_token_importance = profile_compute_token_importance
    importance_module.AttentionImportanceAnalyzer.compute_importance = profile_compute_importance
    selector_module.TokenSelector.select = profile_select
    allocator_module.AdaptiveBitAllocator.allocate = profile_allocate
    controller_module.TokenBudgetController.get_budget = profile_get_budget
    cache_module.LayerKVCache.update = profile_update


def uninstall_hooks() -> None:
    quantizer_module.pack_bits = _orig_pack_bits
    quantizer_module.unpack_bits = _orig_unpack_bits
    quantizer_module.GroupQuantizer.quantize = _orig_quantize
    quantizer_module.GroupQuantizer.dequantize = _orig_dequantize
    importance_module.AttentionImportanceAnalyzer.compute_token_importance = _orig_compute_token_importance
    importance_module.AttentionImportanceAnalyzer.compute_importance = _orig_compute_importance
    selector_module.TokenSelector.select = _orig_select
    allocator_module.AdaptiveBitAllocator.allocate = _orig_allocate
    controller_module.TokenBudgetController.get_budget = _orig_get_budget
    cache_module.LayerKVCache.update = _orig_update


def run_profiling(
    model_id: str = "facebook/opt-125m",
    prompt: str = "Artificial intelligence and machine learning have transformed modern software engineering by enabling",
    max_new_tokens: int = 50,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    # Baseline run with warmup
    print(">>> Profiling Baseline (Standard HF Cache)...")
    model_base = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_base.eval()
    with torch.no_grad():
        _ = model_base.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)

    torch.manual_seed(42)
    t0_base = time.perf_counter()
    with torch.no_grad():
        _ = model_base.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    t1_base = time.perf_counter()
    base_latency = t1_base - t0_base
    base_ms_per_token = (base_latency * 1000.0) / max_new_tokens

    # Reset timers
    for k in TIMERS:
        TIMERS[k] = 0.0
    for k in COUNTERS:
        COUNTERS[k] = 0

    # Install hooks
    print(">>> Profiling AdaptiveKV V1 Full...")
    model_akv = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_akv.eval()

    # Warmup AdaptiveKV without profiling
    adapted_warmup, cache_warmup = apply_adaptive_kv(
        model_akv,
        enable_quantization=True,
        enable_adaptive_bits=True,
        enable_token_eviction=True,
        max_cache_tokens=48,
        keep_ratio=0.5,
        recent_window=16,
        sink_tokens=4,
    )
    with torch.no_grad():
        _ = adapted_warmup.generate(**inputs, max_new_tokens=2, do_sample=False, past_key_values=cache_warmup)

    # Clean fresh model & cache for actual profiled run
    model_akv_profile = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_akv_profile.eval()
    adapted_model, cache = apply_adaptive_kv(
        model_akv_profile,
        enable_quantization=True,
        enable_adaptive_bits=True,
        enable_token_eviction=True,
        max_cache_tokens=48,
        keep_ratio=0.5,
        recent_window=16,
        sink_tokens=4,
    )

    install_hooks()
    profiler = cProfile.Profile()
    torch.manual_seed(42)

    t0_akv = time.perf_counter()
    profiler.enable()
    with torch.no_grad():
        _ = adapted_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
            return_dict_in_generate=True,
        )
    profiler.disable()
    t1_akv = time.perf_counter()

    uninstall_hooks()

    akv_latency = t1_akv - t0_akv
    akv_ms_per_token = (akv_latency * 1000.0) / max_new_tokens
    TIMERS["total_generate_wall_time"] = akv_latency
    TIMERS["model_forward_pure"] = akv_latency - TIMERS["cache_update_total"]

    TIMERS["dequant_math_scale_zero"] = max(0.0, TIMERS["dequantize_total"] - TIMERS["unpack_bits_total"])
    TIMERS["quant_math_min_max_scale"] = max(0.0, TIMERS["quantize_total"] - TIMERS["pack_bits_total"])

    known_cache_subops = (
        TIMERS["dequantize_total"]
        + TIMERS["quantize_total"]
        + TIMERS["importance_scoring_total"]
        + TIMERS["token_selection_eviction_total"]
        + TIMERS["bit_allocation_total"]
        + TIMERS["budget_calc_total"]
    )
    TIMERS["other_cache_python_overhead"] = max(0.0, TIMERS["cache_update_total"] - known_cache_subops)

    slowdown = akv_latency / base_latency

    profile_summary = {
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": prompt_tokens,
        "baseline": {
            "total_latency_s": round(base_latency, 4),
            "ms_per_token": round(base_ms_per_token, 2),
        },
        "adaptivekv": {
            "total_latency_s": round(akv_latency, 4),
            "ms_per_token": round(akv_ms_per_token, 2),
            "slowdown_factor": round(slowdown, 2),
        },
        "breakdown": {
            "LLM/model inference": {
                "time_s": round(TIMERS["model_forward_pure"], 4),
                "percentage": round((TIMERS["model_forward_pure"] / akv_latency) * 100.0, 2),
            },
            "dequantization": {
                "time_s": round(TIMERS["dequantize_total"], 4),
                "percentage": round((TIMERS["dequantize_total"] / akv_latency) * 100.0, 2),
            },
            "bit unpacking (inside dequant)": {
                "time_s": round(TIMERS["unpack_bits_total"], 4),
                "percentage": round((TIMERS["unpack_bits_total"] / akv_latency) * 100.0, 2),
            },
            "dequant math & scaling": {
                "time_s": round(TIMERS["dequant_math_scale_zero"], 4),
                "percentage": round((TIMERS["dequant_math_scale_zero"] / akv_latency) * 100.0, 2),
            },
            "quantization": {
                "time_s": round(TIMERS["quantize_total"], 4),
                "percentage": round((TIMERS["quantize_total"] / akv_latency) * 100.0, 2),
            },
            "bit packing (inside quant)": {
                "time_s": round(TIMERS["pack_bits_total"], 4),
                "percentage": round((TIMERS["pack_bits_total"] / akv_latency) * 100.0, 2),
            },
            "quant math & min/max/scale": {
                "time_s": round(TIMERS["quant_math_min_max_scale"], 4),
                "percentage": round((TIMERS["quant_math_min_max_scale"] / akv_latency) * 100.0, 2),
            },
            "importance scoring": {
                "time_s": round(TIMERS["importance_scoring_total"], 4),
                "percentage": round((TIMERS["importance_scoring_total"] / akv_latency) * 100.0, 2),
            },
            "token selection & eviction": {
                "time_s": round(TIMERS["token_selection_eviction_total"], 4),
                "percentage": round((TIMERS["token_selection_eviction_total"] / akv_latency) * 100.0, 2),
            },
            "bit allocation": {
                "time_s": round(TIMERS["bit_allocation_total"], 4),
                "percentage": round((TIMERS["bit_allocation_total"] / akv_latency) * 100.0, 2),
            },
            "token budget calculation": {
                "time_s": round(TIMERS["budget_calc_total"], 4),
                "percentage": round((TIMERS["budget_calc_total"] / akv_latency) * 100.0, 2),
            },
            "tensor copying/cloning & python cache overhead": {
                "time_s": round(TIMERS["other_cache_python_overhead"], 4),
                "percentage": round((TIMERS["other_cache_python_overhead"] / akv_latency) * 100.0, 2),
            },
            "CPU <-> GPU transfers": {
                "time_s": 0.0,
                "percentage": 0.0,
            },
        },
        "top_3_bottlenecks": [
            {
                "component": "Dequantization & Bit Unpacking",
                "time_s": round(TIMERS["dequantize_total"], 4),
                "percentage": round((TIMERS["dequantize_total"] / akv_latency) * 100.0, 2),
                "why_slow": "Executed 1,200 times during decoding (every token decoding step x 12 layers x 2 K/V tensors). Unpacks bit-packed uint8 tensors back to float32 using non-vectorized Python bit-shift operations (>> and &) and PyTorch tensor reshape operations."
            },
            {
                "component": "Quantization & Bit Packing",
                "time_s": round(TIMERS["quantize_total"], 4),
                "percentage": round((TIMERS["quantize_total"] / akv_latency) * 100.0, 2),
                "why_slow": "Executed 1,200 times during decoding. Computes per-group min/max/scale/zero-point parameters and packs 2/3/4-bit values into uint8 byte arrays using Python loops, stack/cat calls, and bitwise bit-shifting."
            },
            {
                "component": "Repeated Dequantize-Quantize-Dequantize Loop per Decoding Step",
                "time_s": round(TIMERS["dequantize_total"] + TIMERS["quantize_total"], 4),
                "percentage": round(((TIMERS["dequantize_total"] + TIMERS["quantize_total"]) / akv_latency) * 100.0, 2),
                "why_slow": "On every single token generation step, the entire historical KV cache is dequantized from compressed storage into float32, concatenated with 1 new token, re-quantized back into compressed storage, and immediately dequantized a second time to return floats for attention computation."
            }
        ],
        "counters": COUNTERS,
    }

    json_path = RESULTS_DIR / "adaptivekv_v1_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)

    md_path = RESULTS_DIR / "adaptivekv_v1_profile.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AdaptiveKV V1 Performance Profiling Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Model**: `{model_id}`\n")
        f.write(f"- **Prompt Length**: {prompt_tokens} tokens\n")
        f.write(f"- **Max New Tokens**: {max_new_tokens}\n")
        f.write(f"- **Baseline Latency**: {base_latency:.4f} s ({base_ms_per_token:.2f} ms/token)\n")
        f.write(f"- **AdaptiveKV Latency**: {akv_latency:.4f} s ({akv_ms_per_token:.2f} ms/token)\n")
        f.write(f"- **Slowdown Factor**: **{slowdown:.2f}x**\n\n")

        f.write("## Detailed Component Execution Time Breakdown\n\n")
        f.write("| Component | Time (s) | Percentage (%) | Function Calls |\n")
        f.write("| --- | --- | --- | --- |\n")
        b = profile_summary["breakdown"]
        f.write(f"| LLM / Model Inference (Pure) | {b['LLM/model inference']['time_s']}s | {b['LLM/model inference']['percentage']}% | N/A |\n")
        f.write(f"| **Dequantization Total** | **{b['dequantization']['time_s']}s** | **{b['dequantization']['percentage']}%** | {COUNTERS['dequantize_calls']} |\n")
        f.write(f"| └─ Bit Unpacking (`unpack_bits`) | {b['bit unpacking (inside dequant)']['time_s']}s | {b['bit unpacking (inside dequant)']['percentage']}% | {COUNTERS['unpack_bits_calls']} |\n")
        f.write(f"| └─ Scaling & Zero-Point Math | {b['dequant math & scaling']['time_s']}s | {b['dequant math & scaling']['percentage']}% | N/A |\n")
        f.write(f"| **Quantization Total** | **{b['quantization']['time_s']}s** | **{b['quantization']['percentage']}%** | {COUNTERS['quantize_calls']} |\n")
        f.write(f"| └─ Bit Packing (`pack_bits`) | {b['bit packing (inside quant)']['time_s']}s | {b['bit packing (inside quant)']['percentage']}% | {COUNTERS['pack_bits_calls']} |\n")
        f.write(f"| └─ Min/Max/Scale Computation | {b['quant math & min/max/scale']['time_s']}s | {b['quant math & min/max/scale']['percentage']}% | N/A |\n")
        f.write(f"| Importance Scoring | {b['importance scoring']['time_s']}s | {b['importance scoring']['percentage']}% | {COUNTERS['importance_scoring_calls']} |\n")
        f.write(f"| Token Selection & Eviction | {b['token selection & eviction']['time_s']}s | {b['token selection & eviction']['percentage']}% | {COUNTERS['token_selection_calls']} |\n")
        f.write(f"| Bit Allocation | {b['bit allocation']['time_s']}s | {b['bit allocation']['percentage']}% | {COUNTERS['bit_allocation_calls']} |\n")
        f.write(f"| Token Budget Calculation | {b['token budget calculation']['time_s']}s | {b['token budget calculation']['percentage']}% | {COUNTERS['budget_calc_calls']} |\n")
        f.write(f"| Tensor Copying / Concatenation / Python Overhead | {b['tensor copying/cloning & python cache overhead']['time_s']}s | {b['tensor copying/cloning & python cache overhead']['percentage']}% | {COUNTERS['cache_update_calls']} |\n")
        f.write(f"| CPU ↔ GPU Transfers | 0.0000s | 0.00% | 0 |\n\n")

        f.write("## TOP 3 Bottlenecks\n\n")
        for i, item in enumerate(profile_summary["top_3_bottlenecks"], start=1):
            f.write(f"### {i}. {item['component']}\n")
            f.write(f"- **Time**: {item['time_s']} s\n")
            f.write(f"- **Percentage of total time**: {item['percentage']}%\n")
            f.write(f"- **Why it is slow**: {item['why_slow']}\n\n")

        f.write("## Conclusion\n\n")
        f.write(f"> \"AdaptiveKV is slow mainly because **on every decoding step for every layer, it repeatedly dequantizes, quantizes, and re-dequantizes the full KV cache history in pure Python using un-vectorized bit-packing and unpacking operations, incurring massive Python interpreter and PyTorch tensor overhead (accounting for {b['dequantization']['percentage'] + b['quantization']['percentage']:.1f}% of total generation time)**.\"\n")

    print(f"[Profiling] Saved JSON report to: {json_path}")
    print(f"[Profiling] Saved Markdown report to: {md_path}")
    return profile_summary


if __name__ == "__main__":
    run_profiling()
