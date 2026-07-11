from pathlib import Path

from ayn_vqa.stages.select import Prediction
from ayn_vqa.submission import check_submission_format, write_predictions_csv


def test_write_predictions_csv_has_exact_expected_columns(tmp_path: Path) -> None:
    predictions = [
        Prediction(record_id="a", pred=0, confidence=None, raw=None),
        Prediction(record_id="b", pred=2, confidence=0.9, raw="random"),
    ]
    out_path = write_predictions_csv(predictions, tmp_path / "prediction.csv")

    text = out_path.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "id,prediction"
    assert lines[1] == "a,0"
    assert lines[2] == "b,2"


def test_check_submission_format_passes_a_well_formed_csv(tmp_path: Path) -> None:
    predictions = [Prediction(f"id{i}", i % 3, None, None) for i in range(6)]
    csv_path = write_predictions_csv(predictions, tmp_path / "prediction.csv")

    result = check_submission_format(csv_path, gold_ids=[f"id{i}" for i in range(6)])

    assert result.ok
    assert result.errors == []


def test_check_submission_format_flags_duplicate_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "prediction.csv"
    csv_path.write_text("id,prediction\nid0,0\nid0,1\n", encoding="utf-8")

    result = check_submission_format(csv_path)

    assert not result.ok
    assert any("duplicate id" in e for e in result.errors)


def test_check_submission_format_flags_out_of_range_prediction(tmp_path: Path) -> None:
    csv_path = tmp_path / "prediction.csv"
    csv_path.write_text("id,prediction\nid0,5\n", encoding="utf-8")

    result = check_submission_format(csv_path)

    assert not result.ok
    assert any("must be a single integer 0, 1 or 2" in e for e in result.errors)


def test_check_submission_format_flags_missing_ids_against_gold(tmp_path: Path) -> None:
    predictions = [Prediction("id0", 0, None, None)]
    csv_path = write_predictions_csv(predictions, tmp_path / "prediction.csv")

    result = check_submission_format(csv_path, gold_ids=["id0", "id1"])

    assert not result.ok
    assert any("missing from your predictions" in e for e in result.errors)
