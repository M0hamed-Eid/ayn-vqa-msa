"""Generic JSONL artifact cache: `artifacts/<split>/<stage>/<config_key>.jsonl`.

Every stage in this pipeline (ASR now; parsing/evidence/selection in later
milestones) is a function over records that costs real time or money to
run and is cheap to replay from disk. This is the one place that knows how
to read/write that cache, so every stage's runner looks the same: load
what's cached, run only what's missing, append results as they land (so a
run that dies partway through doesn't lose the items it already finished).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def artifact_path(artifacts_dir: Path, split: str, stage: str, config_key: str) -> Path:
    return artifacts_dir / split / stage / f"{config_key}.jsonl"


def read_jsonl_cache(path: Path, id_field: str = "record_id") -> dict[str, dict[str, Any]]:
    """Returns `{row[id_field]: row}`. Missing file -> empty cache, not an
    error -- the normal state before a stage has ever been run for this
    config.

    Defaults to `"record_id"` because every stage dataclass this cache
    stores (`Transcript` now; parse/evidence/select outputs later) uses
    that field name -- the raw dataset's own `Task1aRecord.id` is the one
    exception, and callers caching *that* shape should pass
    `id_field="id"` explicitly.
    """
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row[id_field]] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Appends one row, flushing to disk immediately. Called once per
    record rather than batched, so a long ASR/API run that's interrupted
    keeps every result computed before the interruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
