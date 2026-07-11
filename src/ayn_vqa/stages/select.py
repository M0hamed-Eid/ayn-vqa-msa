"""Answer-selection stage: given a record, produce a predicted option index.

`AnswerSelector` is a `Protocol` so M2+'s real selectors (ASR + joint-MCQ
VLM prompting, entailment ranking, the end-to-end omni model, ...) slot in
next to these two trivial baselines without any caller needing to change.
Nothing in this file touches audio or images -- that's the entire point of
M1: prove the load -> predict -> submit -> score harness works before any
model exists to plug into it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from ayn_vqa.data.schema import Task1aRecord


@dataclass(frozen=True)
class Prediction:
    """One selector's answer for one record.

    `confidence` and `raw` are optional because these two baselines barely
    have either -- they exist so M2+ selectors (which will have a real
    confidence score and a raw model response to log) fit the same shape.
    """

    record_id: str
    pred: int  # 0, 1, or 2
    confidence: float | None
    raw: str | None


class AnswerSelector(Protocol):
    def predict(self, record: Task1aRecord) -> Prediction: ...


class RandomSelector:
    """Uniform random guess over {0, 1, 2}, seeded for reproducibility.

    This is the dataset's chance floor (~33% on a near-balanced 3-way
    task, per docs/M0_DATA_AUDIT.md's label distribution) -- the number
    every real selector built in M3+ has to clear to be worth anything.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def predict(self, record: Task1aRecord) -> Prediction:
        pred = self._rng.randrange(3)
        return Prediction(record_id=record.id, pred=pred, confidence=1 / 3, raw="random")


class ConstantSelector:
    """Always predicts the same option index, regardless of the record.

    Exposes exactly how much accuracy is "for free" from the label
    distribution's mild skew toward one position, rather than genuine
    reasoning -- any real selector that can't beat this on top of
    `RandomSelector` isn't using the image/audio at all.
    """

    def __init__(self, value: int = 0) -> None:
        if value not in (0, 1, 2):
            raise ValueError(f"value must be 0, 1, or 2; got {value!r}")
        self._value = value

    def predict(self, record: Task1aRecord) -> Prediction:
        return Prediction(
            record_id=record.id, pred=self._value, confidence=None, raw=f"constant-{self._value}"
        )


def predict_split(selector: AnswerSelector, records: list[Task1aRecord]) -> list[Prediction]:
    return [selector.predict(record) for record in records]
