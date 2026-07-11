"""Compare cached ASR runs (e.g. `whisper-small` vs. `whisper-medium`)
side by side.

Deliberately separate from `run_asr_bench.py`: that module executes one
backend; this one only ever reads already-cached JSONL and never calls a
model or API, so comparing runs costs nothing and can be re-run freely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ayn_vqa.artifacts import read_jsonl_cache


def compare_asr_runs(cache_paths: dict[str, Path]) -> pd.DataFrame:
    """One row per (backend_label, record_id): transcript, character
    count, latency, success -- long-format so records missing from one
    backend's cache don't force awkward NaN-filled columns.
    """
    rows: list[dict[str, Any]] = []
    for label, path in cache_paths.items():
        for record_id, row in read_jsonl_cache(path).items():
            text = row.get("text")
            rows.append(
                {
                    "backend": label,
                    "id": record_id,
                    "text": text,
                    "chars": len(text) if text else 0,
                    "latency_sec": row.get("latency_sec"),
                    "ok": row.get("error") is None,
                }
            )
    return pd.DataFrame(rows, columns=["backend", "id", "text", "chars", "latency_sec", "ok"])


def summarize_by_backend(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per-backend aggregate: success rate, mean/median length, mean latency."""
    if comparison.empty:
        return pd.DataFrame(
            columns=["backend", "n", "n_ok", "mean_chars", "median_chars", "mean_latency_sec"]
        )

    ok = comparison[comparison["ok"]]
    summary = (
        ok.groupby("backend")
        .agg(
            n_ok=("id", "count"),
            mean_chars=("chars", "mean"),
            median_chars=("chars", "median"),
            mean_latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )
    totals = comparison.groupby("backend")["id"].count().rename("n")
    return summary.merge(totals, on="backend").sort_values("backend").reset_index(drop=True)
