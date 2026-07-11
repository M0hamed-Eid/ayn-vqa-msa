"""CSV writer for Task 1a predictions, plus an auto-run of the vendored
format checker.

Writing predictions and immediately checking their format in the same
call is deliberate: a submission bug caught locally costs nothing; the
same bug caught by Codabench costs one of a small number of daily
submission attempts.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ayn_vqa.config import PROJECT_ROOT
from ayn_vqa.stages.select import Prediction

_FORMAT_CHECKER_DIR = PROJECT_ROOT / "scorer_official" / "format_checker"

_CheckFn = Callable[
    [list[dict[str, Any]], "list[str] | None", "list[str] | None"], tuple[list[str], list[str]]
]


def _vendored_check_1a() -> _CheckFn:
    """Imported lazily (only when actually needed) rather than at module
    import time, so merely importing `submission` doesn't mutate
    `sys.path` as a side effect.
    """
    import sys

    if str(_FORMAT_CHECKER_DIR) not in sys.path:
        sys.path.insert(0, str(_FORMAT_CHECKER_DIR))
    import check_format  # type: ignore[import-not-found]  # vendored, see scorer_official/README.md

    fn: _CheckFn = check_format.check_1a
    return fn


@dataclass(frozen=True)
class FormatCheckResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def write_predictions_csv(predictions: list[Prediction], out_path: Path) -> Path:
    """Writes exactly the columns the scorer and Codabench expect:
    `id,prediction` -- see scorer_official/format_checker/README.md.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction"])
        for prediction in predictions:
            writer.writerow([prediction.record_id, prediction.pred])
    return out_path


def check_submission_format(csv_path: Path, gold_ids: list[str] | None = None) -> FormatCheckResult:
    """Runs the vendored `check_1a` (not just `write_predictions_csv`'s own
    logic) so "does this pass the official checker" is answered by the
    actual checker, not by our belief that our writer is correct.
    """
    check_1a = _vendored_check_1a()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else None

    errors, warnings = check_1a(rows, fieldnames, gold_ids)
    return FormatCheckResult(errors=errors, warnings=warnings)
