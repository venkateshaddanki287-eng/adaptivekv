"""Streamlit research dashboard for AdaptiveKV benchmark analysis."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import pandas as pd
    import streamlit as st
except ImportError:
    st = None

RESULTS_DIR = Path(__file__).parent.parent / "research" / "results"


def load_data() -> list[dict]:
    records = []
    if RESULTS_DIR.exists():
        for json_file in sorted(RESULTS_DIR.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        records.append(data)
            except Exception:
                pass
    return records


def main() -> None:
    if st is None:
        print("[Dashboard] Streamlit not installed. Launch local dashboard using: python dashboard/server.py")
        return

    st.set_page_config(page_title="AdaptiveKV Research Dashboard", layout="wide")
    st.title("⚡ AdaptiveKV Research Dashboard")
    st.caption("Empirical KV-Cache Compression Analytics & Dynamic Bit Allocation")

    records = load_data()
    if not records:
        st.warning("No benchmark result JSON files found in research/results/. Run `adaptivekv benchmark` first.")
        return

    df = pd.DataFrame(records)
    st.sidebar.header("Filter Results")
    selected_ctx = st.sidebar.selectbox("Context Length", ["All"] + sorted(df["context_length"].unique().tolist()))
    selected_method = st.sidebar.selectbox("Compression Method", ["All"] + sorted(df["method"].unique().tolist()))

    filtered_df = df.copy()
    if selected_ctx != "All":
        filtered_df = filtered_df[filtered_df["context_length"] == selected_ctx]
    if selected_method != "All":
        filtered_df = filtered_df[filtered_df["method"] == selected_method]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Compression Ratio", f"{filtered_df['compression_ratio'].max():.2f}x")
    with col2:
        st.metric("Max Memory Saved", f"{filtered_df['memory_saved_percent'].max():.1f}%")
    with col3:
        st.metric("Best Cosine Similarity", f"{filtered_df['cosine_similarity'].max():.4f}")
    with col4:
        st.metric("Min MSE", f"{filtered_df['mse'].min():.6f}")

    st.subheader("Raw Benchmark Dataset")
    st.dataframe(filtered_df)


if __name__ == "__main__":
    main()
