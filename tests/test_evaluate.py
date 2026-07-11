"""Confirms our wrapper reproduces a hand-computed accuracy/balanced-accuracy
/macro-F1 on a small fixture -- exactly the M1 acceptance test from the
project's roadmap ("scorer reproduces hand-computed accuracy on 10 items").

Ground truth (10 items): labels are 0,0,0,1,1,1,2,2,2,2 (3/3/4 split).
Predictions get 7/10 right, with a specific wrong pattern chosen so
accuracy, balanced accuracy, and macro-F1 all differ from each other --
if any were miscomputed, at least one assertion would catch it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.evaluate import score_predictions

_GOLD_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
# id2, id4, id8 are wrong (predict a neighboring class); everything else correct.
_PRED = {
    "id0": 0,
    "id1": 0,
    "id2": 1,
    "id3": 1,
    "id4": 0,
    "id5": 1,
    "id6": 2,
    "id7": 2,
    "id8": 0,
    "id9": 2,
}


def _gold_records() -> list[Task1aRecord]:
    return [
        Task1aRecord.model_validate(
            {
                "id": f"id{i}",
                "image": f"images/id{i}.jpg",
                "audio": f"audio/msa/id{i}.wav",
                "label": label,
            }
        )
        for i, label in enumerate(_GOLD_LABELS)
    ]


@pytest.fixture
def pred_csv(tmp_path: Path) -> Path:
    path = tmp_path / "prediction.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("id,prediction\n")
        for record_id, pred in _PRED.items():
            f.write(f"{record_id},{pred}\n")
    return path


def test_score_predictions_matches_hand_computed_metrics(pred_csv: Path) -> None:
    metrics = score_predictions(_gold_records(), pred_csv)
    by_name = {m.name: m.value for m in metrics}

    # accuracy: 7 correct / 10 total
    assert by_name["accuracy"] == pytest.approx(0.7, abs=1e-6)
    # balanced_accuracy: mean per-class recall = (2/3 + 2/3 + 3/4) / 3 = 25/36
    assert by_name["balanced_accuracy"] == pytest.approx(25 / 36, abs=1e-6)
    # macro_f1: mean per-class F1 = (4/7 + 2/3 + 6/7) / 3 = 44/63
    assert by_name["macro_f1"] == pytest.approx(44 / 63, abs=1e-6)

    ranking = [m for m in metrics if m.is_ranking]
    assert [m.name for m in ranking] == ["accuracy"]


def test_score_predictions_raises_runtime_error_on_unlabeled_gold(pred_csv: Path) -> None:
    devtest_style_records = [
        Task1aRecord.model_validate(
            {"id": "id0", "image": "images/id0.jpg", "audio": "audio/msa/id0.wav"}
        )
    ]

    with pytest.raises(RuntimeError, match="labeled split"):
        score_predictions(devtest_style_records, pred_csv)
