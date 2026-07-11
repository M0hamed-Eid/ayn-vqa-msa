from pathlib import Path

from ayn_vqa.artifacts import append_jsonl
from ayn_vqa.asr_compare import compare_asr_runs, summarize_by_backend


def test_compare_and_summarize_two_runs(tmp_path: Path) -> None:
    small_path = tmp_path / "small.jsonl"
    medium_path = tmp_path / "medium.jsonl"

    append_jsonl(small_path, {"record_id": "a", "text": "short", "latency_sec": 1.0, "error": None})
    append_jsonl(
        small_path, {"record_id": "b", "text": "also short", "latency_sec": 1.2, "error": None}
    )
    append_jsonl(
        medium_path,
        {"record_id": "a", "text": "a longer transcript", "latency_sec": 3.0, "error": None},
    )
    append_jsonl(
        medium_path, {"record_id": "b", "text": None, "latency_sec": None, "error": "timeout"}
    )

    comparison = compare_asr_runs({"whisper-small": small_path, "whisper-medium": medium_path})

    assert len(comparison) == 4
    assert set(comparison["backend"]) == {"whisper-small", "whisper-medium"}
    assert comparison.loc[comparison["id"] == "a", "chars"].tolist() == [5, 19]

    summary = summarize_by_backend(comparison)
    by_backend = summary.set_index("backend")

    assert by_backend.loc["whisper-small", "n"] == 2
    assert by_backend.loc["whisper-small", "n_ok"] == 2
    assert by_backend.loc["whisper-medium", "n"] == 2
    assert by_backend.loc["whisper-medium", "n_ok"] == 1  # "b" had an error


def test_summarize_by_backend_handles_empty_input() -> None:
    import pandas as pd

    empty = pd.DataFrame(columns=["backend", "id", "text", "chars", "latency_sec", "ok"])
    summary = summarize_by_backend(empty)
    assert summary.empty
