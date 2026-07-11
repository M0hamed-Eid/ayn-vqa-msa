from __future__ import annotations

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.error_analysis import build_error_table, summarize_by


def _record(record_id: str, country: str, category: str, label: int) -> Task1aRecord:
    return Task1aRecord.model_validate(
        {
            "id": record_id,
            "image": f"images/{record_id}.jpg",
            "audio": f"audio/msa/{record_id}.wav",
            "country": country,
            "category": category,
            "label": label,
        }
    )


def test_build_error_table_marks_correctness() -> None:
    records = [
        _record("a", "Egypt", "Food", 0),
        _record("b", "Egypt", "Food", 1),
        _record("c", "Jordan", "Sports", 2),
    ]
    transcripts = {"a": {"text": "t-a"}, "b": {"text": "t-b"}, "c": {"error": "asr failed"}}
    parses = {"a": {"question": "q-a"}, "b": {"question": "q-b"}, "c": {"error": "no transcript"}}
    predictions = {"a": 0, "b": 0, "c": 2}  # a correct, b wrong, c correct

    table = build_error_table(records, transcripts, parses, predictions).set_index("id")

    assert bool(table.loc["a", "correct"]) is True
    assert bool(table.loc["b", "correct"]) is False
    assert bool(table.loc["c", "correct"]) is True
    assert table.loc["a", "transcript"] == "t-a"
    assert table.loc["c", "asr_error"] == "asr failed"


def test_summarize_by_country_computes_accuracy() -> None:
    records = [
        _record("a", "Egypt", "Food", 0),
        _record("b", "Egypt", "Food", 1),
        _record("c", "Jordan", "Sports", 2),
    ]
    table = build_error_table(records, {}, {}, {"a": 0, "b": 0, "c": 2})

    by_country = summarize_by(table, "country").set_index("country")

    assert by_country.loc["Egypt", "n"] == 2
    assert by_country.loc["Egypt", "n_correct"] == 1
    assert by_country.loc["Egypt", "accuracy"] == 0.5
    assert by_country.loc["Jordan", "accuracy"] == 1.0


def test_summarize_by_handles_no_labeled_rows() -> None:
    records = [
        Task1aRecord.model_validate(
            {"id": "a", "image": "images/a.jpg", "audio": "audio/msa/a.wav"}
        )
    ]
    table = build_error_table(records, {}, {}, {"a": 0})

    summary = summarize_by(table, "country")

    assert summary.empty
