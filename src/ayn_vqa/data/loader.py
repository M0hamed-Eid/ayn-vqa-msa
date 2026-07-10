"""Load `task1a/<split>_<language>.jsonl` into validated records, tolerant
of malformed rows.

A data-audit tool that crashes on the first bad line defeats its own
purpose -- the point of M0 is to surface every problem in one pass. Every
public function here returns the records that parsed *and* a list of the
ones that didn't, instead of raising past the first failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ayn_vqa.data.schema import Language, Split, Task1aRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordError:
    """One JSONL line that failed to parse as JSON or validate as a record."""

    line_number: int
    raw_line: str
    message: str


@dataclass(frozen=True)
class SplitData:
    """Everything loaded from one `<split>_<language>.jsonl` file."""

    task: str
    split: Split
    language: Language
    jsonl_path: Path
    records: list[Task1aRecord] = field(default_factory=list)
    errors: list[RecordError] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)


def jsonl_path_for(data_root: Path, task: str, split: Split, language: Language) -> Path:
    """Reproduces the naming convention from the dataset README:
    `<task>/<split>_<language>.jsonl`."""
    return data_root / task / f"{split.value}_{language.value}.jsonl"


def load_split(
    data_root: Path,
    split: Split,
    language: Language,
    task: str = "task1a",
) -> SplitData:
    """Parse one JSONL split file line by line.

    Deliberately not `pandas.read_json(lines=True)` or `datasets.load_dataset`:
    both raise on the first malformed line, or bury which line was bad behind
    a generic parser error. A plain loop costs nothing at this dataset's
    scale (thousands of lines) and gives every failure a line number to
    act on.
    """
    path = jsonl_path_for(data_root, task, split, language)
    if not path.exists():
        raise FileNotFoundError(
            f"split file not found: {path} "
            f"(task={task!r}, split={split.value!r}, language={language.value!r})"
        )

    records: list[Task1aRecord] = []
    errors: list[RecordError] = []

    with path.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                record = Task1aRecord.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(RecordError(line_number, stripped, str(exc)))
                continue
            records.append(record)

    if errors:
        logger.warning(
            "%s: %d/%d lines failed to parse/validate",
            path,
            len(errors),
            len(errors) + len(records),
        )

    return SplitData(
        task=task,
        split=split,
        language=language,
        jsonl_path=path,
        records=records,
        errors=errors,
    )


def load_all_splits(
    data_root: Path,
    splits: tuple[Split, ...] = (Split.TRAIN, Split.DEV, Split.DEVTEST),
    language: Language = Language.MSA,
    task: str = "task1a",
) -> dict[Split, SplitData]:
    """Convenience wrapper: load every split of one language track."""
    return {split: load_split(data_root, split, language, task=task) for split in splits}
