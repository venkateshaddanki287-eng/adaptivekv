"""Generate research comparison tables from empirical raw JSON records."""

from __future__ import annotations

import json
from pathlib import Path

RAW_FILE = Path("research/results/research_experiment_raw.json")
TABLE_DIR = Path("research/tables")


def generate_tables() -> None:
    if not RAW_FILE.exists():
        print(f"[Error] {RAW_FILE} not found.")
        return

    with open(RAW_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("[Warning] No results to format.")
        return

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Main Performance & Quality Table (Markdown & CSV)
    md_lines = [
        "# Table 1: Comprehensive Empirical Comparison of AdaptiveKV vs. Baselines",
        "",
        "| Context (Tokens) | Compression Scheme | Storage (KB) | Comp Ratio | Memory Saved (%) | Cosine Sim (Quality) | Token Agreement (%) | Total Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    csv_lines = [
        "Context,Scheme,Storage_KB,Compression_Ratio,Memory_Saved_Pct,Cosine_Similarity,Token_Agreement_Pct,Latency_ms"
    ]

    for r in results:
        ctx = r["context_length"]
        m = r["method"]
        bytes_kb = r["compressed_bytes"] / 1024.0 if "compressed_bytes" in r else r.get("compressed_bytes_mean", 0)/1024.0
        comp = r.get("compression_ratio", r.get("compression_ratio_mean", 1.0))
        saved = r.get("memory_saved_percent", r.get("memory_saved_percent_mean", 0.0))
        cos = r.get("cosine_similarity", r.get("cosine_similarity_mean", 1.0))
        agree = r.get("token_agreement_percent", r.get("token_agreement_mean_pct", 100.0))
        lat = r.get("latency_ms", r.get("latency_mean_ms", 0.0))

        md_lines.append(f"| {ctx} | **{m}** | {bytes_kb:,.1f} | {comp:.2f}x | {saved:.1f}% | {cos:.4f} | {agree:.1f}% | {lat:.1f} |")
        csv_lines.append(f"{ctx},{m},{bytes_kb:.1f},{comp:.2f},{saved:.1f},{cos:.4f},{agree:.1f},{lat:.1f}")

    (TABLE_DIR / "table1_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    (TABLE_DIR / "table1_comparison.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    print(f"[Success] Generated research tables in {TABLE_DIR}/")


if __name__ == "__main__":
    generate_tables()
