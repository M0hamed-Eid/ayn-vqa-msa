"""Thin wrapper around the vendored official scorer.

We do not reimplement accuracy/balanced-accuracy/macro-F1 -- we import the
organizers' exact `score_1a` function unchanged (see
`scorer_official/README.md`), so a local dev score can never silently
drift from what the Codabench leaderboard actually computes.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ayn_vqa.config import PROJECT_ROOT
from ayn_vqa.data.schema import Task1aRecord

_SCORER_DIR = PROJECT_ROOT / "scorer_official" / "scorer"

_ScoreFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[tuple[str, float, bool]]]


def _vendored_score_1a() -> _ScoreFn:
    """Imported lazily -- see `submission._vendored_check_format` for why."""
    import sys

    if str(_SCORER_DIR) not in sys.path:
        sys.path.insert(0, str(_SCORER_DIR))
    import score  # type: ignore[import-not-found]  # vendored, see scorer_official/README.md

    fn: _ScoreFn = score.score_1a
    return fn


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    is_ranking: bool


def _read_pred_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def score_predictions(gold_records: list[Task1aRecord], pred_csv: Path) -> list[Metric]:
    """Scores `pred_csv` against already-loaded, already-validated gold
    records (`train`/`dev` -- `devtest`/`test` have no `label` to score
    against).

    Deliberately takes `Task1aRecord`s, not a gold JSONL path: re-parsing
    the raw file here would duplicate (and, worse, silently diverge from)
    `data/loader.py`'s tolerant parsing -- e.g. a malformed line that the
    loader correctly quarantines as a `RecordError` would otherwise crash
    a naive second JSON parser reading the same file.

    The vendored `score_1a` only checks that a `"label"` *key* is present,
    not that its value is non-`None` -- so an unlabeled (`devtest`/`test`)
    record would silently become the string class `"None"` instead of
    raising, corrupting the metrics rather than failing loudly. We check
    that ourselves before ever calling into the vendored code.

    Separately, `score_1a` calls `sys.exit(1)` on other malformed input
    (e.g. a prediction CSV missing its columns) rather than raising --
    fine for its intended use as a standalone CLI script, but would
    otherwise kill our whole process. We convert that into a normal,
    catchable `RuntimeError` too.
    """
    if not gold_records or any(record.label is None for record in gold_records):
        raise RuntimeError(
            "score_predictions requires a labeled split -- every gold record "
            "needs a `label` (use train/dev, not devtest/test)."
        )

    score_1a = _vendored_score_1a()
    gold_rows: list[dict[str, Any]] = [
        {"id": record.id, "label": record.label} for record in gold_records
    ]
    pred_rows = _read_pred_csv(pred_csv)

    try:
        raw = score_1a(gold_rows, pred_rows)
    except SystemExit as exc:
        raise RuntimeError(
            "official scorer rejected the input -- confirm `gold_records` come "
            "from a labeled split (train/dev, not devtest/test) and `pred_csv` "
            f"has `id,prediction` columns (exit code {exc.code})"
        ) from exc

    return [
        Metric(name=name, value=value, is_ranking=is_ranking) for name, value, is_ranking in raw
    ]
