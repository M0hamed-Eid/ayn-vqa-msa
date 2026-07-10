"""Task 1a's on-disk schema, as a Pydantic model.

The HF dataset card documents this schema in prose; this module is the one
place that turns it into something every later stage (loader, validator,
and eventually the ASR/parsing/VLM stages) imports and trusts, instead of
every stage re-parsing raw dict keys by hand and slowly drifting apart.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class Language(StrEnum):
    """Audio/question language track.

    Images and labels are identical across tracks (the README calls them
    "parallel"); only the audio directory differs. The MSA track is this
    project's competition target -- English audio is present on disk but is
    not a legal input to the scored pipeline, only useful for our own data
    audit/comparison.
    """

    EN = "en"
    MSA = "msa"


class Split(StrEnum):
    """`test` is deliberately not modeled: it doesn't exist on disk yet
    (blind set, released 2026-07-20) and loading it is not an M0 concern.
    """

    TRAIN = "train"
    DEV = "dev"
    DEVTEST = "devtest"


class Task1aRecord(BaseModel):
    """One row of `task1a/<split>_<language>.jsonl`.

    `label`, `country`, `category`, and `subcategory` are only present in
    `train`/`dev`; `devtest` rows omit them entirely (the key is absent, not
    set to `null`), so they default to `None` here rather than being
    required. `extra="forbid"` is intentional: every key set on disk was
    verified during the M0 audit (see docs/M0_DATA_AUDIT.md); a surprise
    field on the eventual blind `test` split should fail loudly here rather
    than be silently dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    image: str
    audio: str
    label: int | None = None
    country: str | None = None
    category: str | None = None
    subcategory: str | None = None

    @field_validator("label")
    @classmethod
    def _label_is_a_valid_option_index(cls, value: int | None) -> int | None:
        if value is not None and value not in (0, 1, 2):
            raise ValueError(f"label must be 0, 1, or 2 (a 3-way MCQ index); got {value!r}")
        return value

    def image_path(self, data_root: Path) -> Path:
        """`image` is stored relative to the dataset root (e.g.
        `images/<id>.jpg`), not relative to `task1a/`."""
        return data_root / self.image

    def audio_path(self, data_root: Path) -> Path:
        """Same convention as `image_path`, e.g. `audio/msa/<id>.wav`."""
        return data_root / self.audio
