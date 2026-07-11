"""End-to-end test of the M2 harness: sample -> transcribe -> cache,
against the synthetic `mini_dataset` fixture, using a fake ASR backend (no
real Whisper model, no network) so it runs instantly.
"""

from __future__ import annotations

from pathlib import Path

from ayn_vqa.data.schema import Language, Split
from ayn_vqa.run_asr_bench import run_asr_bench
from ayn_vqa.stages.asr import Transcript


class _FakeASRBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript:
        self.calls.append(record_id)
        return Transcript(
            record_id, f"transcript for {record_id}", self.name, "fake-model", "msa", 0.01, None
        )


def test_run_asr_bench_transcribes_and_caches(mini_dataset: Path, tmp_path: Path) -> None:
    backend = _FakeASRBackend()

    cache_path = run_asr_bench(
        data_root=mini_dataset,
        artifacts_dir=tmp_path / "artifacts",
        split=Split.TRAIN,
        language=Language.MSA,
        config_key="fake-model",
        backend=backend,
        sample_n=100,  # more than available -> takes all 7 valid records
        seed=42,
    )

    assert cache_path.exists()
    assert len(backend.calls) == 7  # all 7 valid train_msa records (see conftest.mini_dataset)
    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 7


def test_run_asr_bench_reuses_cache_on_second_run(mini_dataset: Path, tmp_path: Path) -> None:
    backend = _FakeASRBackend()
    artifacts_dir = tmp_path / "artifacts"

    run_asr_bench(
        data_root=mini_dataset,
        artifacts_dir=artifacts_dir,
        split=Split.TRAIN,
        language=Language.MSA,
        config_key="fake-model",
        backend=backend,
        sample_n=3,
        seed=42,
    )
    first_call_count = len(backend.calls)
    assert first_call_count == 3

    run_asr_bench(
        data_root=mini_dataset,
        artifacts_dir=artifacts_dir,
        split=Split.TRAIN,
        language=Language.MSA,
        config_key="fake-model",
        backend=backend,
        sample_n=3,
        seed=42,
    )

    # same seed -> same 3 sampled records -> all served from cache, no new calls
    assert len(backend.calls) == first_call_count
