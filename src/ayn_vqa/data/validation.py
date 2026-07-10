"""Does every image/audio path a record points to actually resolve to a
real file on disk?

This is the check the whole M0 milestone exists to run: a cascaded pipeline
that silently skips a missing file, or crashes deep inside an ASR call three
milestones from now, is far worse than a report that says "12 files
missing, here they are" before any model is ever invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ayn_vqa.data.loader import SplitData
from ayn_vqa.data.schema import Task1aRecord

MediaField = Literal["image", "audio"]


@dataclass(frozen=True)
class MediaCheck:
    """Result of checking one record's one media field."""

    record_id: str
    split: str
    language: str
    field: MediaField
    relative_path: str
    absolute_path: Path
    exists: bool
    size_bytes: int | None


def check_record_media(
    record: Task1aRecord, data_root: Path, split: str, language: str
) -> list[MediaCheck]:
    def _check(field_name: MediaField, relative_path: str, absolute_path: Path) -> MediaCheck:
        exists = absolute_path.is_file()
        size = absolute_path.stat().st_size if exists else None
        return MediaCheck(
            record_id=record.id,
            split=split,
            language=language,
            field=field_name,
            relative_path=relative_path,
            absolute_path=absolute_path,
            exists=exists,
            size_bytes=size,
        )

    return [
        _check("image", record.image, record.image_path(data_root)),
        _check("audio", record.audio, record.audio_path(data_root)),
    ]


def validate_split_media(split_data: SplitData, data_root: Path) -> list[MediaCheck]:
    checks: list[MediaCheck] = []
    for record in split_data.records:
        checks.extend(
            check_record_media(record, data_root, split_data.split.value, split_data.language.value)
        )
    return checks


def missing_media(checks: list[MediaCheck]) -> list[MediaCheck]:
    return [c for c in checks if not c.exists]
