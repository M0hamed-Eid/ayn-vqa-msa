"""Per-record error analysis: join transcript, parse, and prediction
output with the gold label into one table, plus per-country/per-category
accuracy breakdowns.

A pure function over already-computed data, not a pipeline stage of its
own -- error analysis should never require a fresh ASR/VLM call, only
reading what the pipeline already cached. This is what lets you answer
"which stage killed this item" (ASR error? parse error? or the VLM just
picked wrong?) by reading one row, instead of cross-referencing three
separate JSONL files by hand.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ayn_vqa.data.schema import Task1aRecord


def build_error_table(
    records: list[Task1aRecord],
    transcripts: dict[str, dict[str, Any]],
    parses: dict[str, dict[str, Any]],
    predictions: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        transcript = transcripts.get(record.id, {})
        parsed = parses.get(record.id, {})
        pred = predictions.get(record.id)
        correct = (
            (pred == record.label) if (pred is not None and record.label is not None) else None
        )
        rows.append(
            {
                "id": record.id,
                "country": record.country,
                "category": record.category,
                "label": record.label,
                "prediction": pred,
                "correct": correct,
                "transcript": transcript.get("text"),
                "asr_error": transcript.get("error"),
                "question": parsed.get("question"),
                "options": parsed.get("options"),
                "parse_error": parsed.get("error"),
            }
        )
    return pd.DataFrame(rows)


def summarize_by(error_table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Accuracy grouped by `column` (e.g. `"country"`, `"category"`),
    restricted to rows that actually have both a prediction and a gold
    label -- unlabeled splits (`devtest`/`test`) correctly produce an
    empty summary rather than a misleading one.
    """
    labeled = error_table.dropna(subset=["correct"])
    if labeled.empty:
        return pd.DataFrame(columns=[column, "n", "n_correct", "accuracy"])
    grouped = labeled.groupby(column)["correct"].agg(n="count", n_correct="sum").reset_index()
    grouped["accuracy"] = grouped["n_correct"] / grouped["n"]
    return grouped.sort_values("accuracy").reset_index(drop=True)
