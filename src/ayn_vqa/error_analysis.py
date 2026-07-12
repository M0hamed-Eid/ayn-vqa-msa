"""Per-record error analysis: join transcript, parse, and prediction
output with the gold label into one table, plus per-country/per-category
accuracy breakdowns.

A pure function over already-computed data, not a pipeline stage of its
own -- error analysis should never require a fresh ASR/VLM call, only
reading what the pipeline already cached. This is what lets you answer
"which stage killed this item" (ASR error? parse error? or the VLM just
picked wrong?) by reading one row, instead of cross-referencing three
separate JSONL files by hand.

`asr_error`/`parse_error` only catch *hard* failures (an exception, an
empty transcript). Hand-reading all 142 wrong M3 dev predictions found
that neither flag catches a third, larger failure mode: the parser's
schema forces exactly three non-empty options every time, so when the
transcript doesn't actually contain three cleanly recoverable ones it
fabricates content instead of failing -- both flags read `None` on those
rows even though the item was never a fair test of the VLM. 69 of the 142
(48.6%) turned out to be exactly this. `option_quality_ok`/
`option_quality_reasons` (from `ayn_vqa.option_quality`) is the column
that actually answers "was this a genuine visual-reasoning miss."
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.option_quality import check_option_quality, check_repetition_loop


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

        options = parsed.get("options")
        option_quality = check_option_quality(tuple(options)) if options else None
        transcript_text = transcript.get("text")
        repetition_loop = transcript_text is not None and check_repetition_loop(transcript_text)

        rows.append(
            {
                "id": record.id,
                "country": record.country,
                "category": record.category,
                "label": record.label,
                "prediction": pred,
                "correct": correct,
                "transcript": transcript_text,
                "asr_error": transcript.get("error"),
                "asr_repetition_loop": repetition_loop,
                "question": parsed.get("question"),
                "options": options,
                "parse_error": parsed.get("error"),
                "option_quality_ok": (
                    None if option_quality is None else not option_quality.is_degenerate
                ),
                "option_quality_reasons": (
                    () if option_quality is None else option_quality.reasons
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_option_quality(error_table: pd.DataFrame) -> dict[str, int]:
    """Among wrong predictions with both a prediction and a gold label:
    how many are pipeline artifacts (degenerate parsed options) vs.
    genuine VLM visual-reasoning misses. The number this module exists to
    surface -- see module docstring.
    """
    wrong = error_table[error_table["correct"].eq(False)]
    n_wrong = len(wrong)
    n_pipeline_artifact = int(wrong["option_quality_ok"].eq(False).sum())
    return {
        "n_wrong": n_wrong,
        "n_pipeline_artifact": n_pipeline_artifact,
        "n_genuine_miss": n_wrong - n_pipeline_artifact,
    }


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
