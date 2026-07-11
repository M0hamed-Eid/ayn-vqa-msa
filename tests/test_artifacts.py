from pathlib import Path

from ayn_vqa.artifacts import append_jsonl, artifact_path, read_jsonl_cache


def test_artifact_path_layout(tmp_path: Path) -> None:
    path = artifact_path(tmp_path, "dev", "asr", "msa_whisper-small")
    assert path == tmp_path / "dev" / "asr" / "msa_whisper-small.jsonl"


def test_read_jsonl_cache_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    cache = read_jsonl_cache(tmp_path / "does-not-exist.jsonl")
    assert cache == {}


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_jsonl(path, {"record_id": "a", "text": "hello", "latency_sec": 1.5})
    append_jsonl(path, {"record_id": "b", "text": "world", "latency_sec": None})

    cache = read_jsonl_cache(path)

    assert set(cache.keys()) == {"a", "b"}
    assert cache["a"]["text"] == "hello"
    assert cache["b"]["latency_sec"] is None


def test_append_jsonl_preserves_non_ascii_text(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_jsonl(path, {"record_id": "a", "text": "مرحبا بالعالم"})

    cache = read_jsonl_cache(path)

    assert cache["a"]["text"] == "مرحبا بالعالم"


def test_read_jsonl_cache_accepts_a_custom_id_field(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_jsonl(path, {"id": "raw-record-id", "image": "images/x.jpg"})

    cache = read_jsonl_cache(path, id_field="id")

    assert set(cache.keys()) == {"raw-record-id"}
