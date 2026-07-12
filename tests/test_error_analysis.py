from __future__ import annotations

from typing import Any

from ayn_vqa.data.schema import Task1aRecord
from ayn_vqa.error_analysis import build_error_table, summarize_by, summarize_option_quality


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


def test_build_error_table_flags_degenerate_options() -> None:
    records = [
        _record("a", "Egypt", "Food", 0),  # clean options, correct
        _record("b", "Egypt", "Food", 1),  # degenerate options, wrong
        _record("c", "Jordan", "Sports", 2),  # no parse at all, wrong
    ]
    transcripts: dict[str, dict[str, Any]] = {
        "a": {"text": "t-a"},
        "b": {"text": "t-b"},
        "c": {"text": "t-c"},
    }
    parses: dict[str, dict[str, Any]] = {
        "a": {"question": "q-a", "options": ("خيار أول", "خيار ثاني", "خيار ثالث")},
        "b": {"question": "q-b", "options": ("خيار أول", "خيار ثاني", "الخيارات هي")},
        "c": {"error": "parse failed"},
    }
    predictions = {"a": 0, "b": 0, "c": 0}  # a correct, b wrong, c wrong

    table = build_error_table(records, transcripts, parses, predictions).set_index("id")

    assert bool(table.loc["a", "option_quality_ok"]) is True
    assert table.loc["a", "option_quality_reasons"] == ()

    assert bool(table.loc["b", "option_quality_ok"]) is False
    b_reasons = table.loc["b", "option_quality_reasons"]
    assert isinstance(b_reasons, tuple)
    assert "boilerplate_leak" in b_reasons

    assert table.loc["c", "option_quality_ok"] is None
    assert table.loc["c", "option_quality_reasons"] == ()


def test_build_error_table_flags_asr_repetition_loop() -> None:
    records = [_record("a", "Egypt", "Food", 0)]
    transcripts: dict[str, dict[str, Any]] = {"a": {"text": "سؤال " + "آ " * 6 + "خيارات"}}
    clean_options = ("خيار أول", "خيار ثاني", "خيار ثالث")
    parses: dict[str, dict[str, Any]] = {"a": {"question": "q", "options": clean_options}}

    table = build_error_table(records, transcripts, parses, {"a": 0}).set_index("id")

    assert bool(table.loc["a", "asr_repetition_loop"]) is True


def test_summarize_option_quality_splits_pipeline_from_genuine_misses() -> None:
    records = [
        _record("a", "Egypt", "Food", 0),  # correct
        _record("b", "Egypt", "Food", 1),  # wrong, degenerate options -> pipeline artifact
        _record("c", "Jordan", "Sports", 2),  # wrong, clean options -> genuine miss
    ]
    transcripts: dict[str, dict[str, Any]] = {"a": {}, "b": {}, "c": {}}
    parses: dict[str, dict[str, Any]] = {
        "a": {"question": "q-a", "options": ("خيار أول", "خيار ثاني", "خيار ثالث")},
        "b": {"question": "q-b", "options": ("خيار أول", "خيار ثاني", "الخيارات هي")},
        "c": {"question": "q-c", "options": ("خيار أول", "خيار ثاني", "خيار ثالث")},
    }
    predictions = {"a": 0, "b": 0, "c": 0}  # a correct, b wrong, c wrong

    table = build_error_table(records, transcripts, parses, predictions)
    summary = summarize_option_quality(table)

    assert summary == {"n_wrong": 2, "n_pipeline_artifact": 1, "n_genuine_miss": 1}
