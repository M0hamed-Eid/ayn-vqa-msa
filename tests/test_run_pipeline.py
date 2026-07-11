"""End-to-end test of the M3 cascade: ASR -> parse -> select -> submit ->
score -> error analysis, against the synthetic `mini_dataset` fixture,
using fake stages (no real Whisper, no real Ollama) so it runs instantly.
"""

from __future__ import annotations

from pathlib import Path

from ayn_vqa.data.schema import Language, Split
from ayn_vqa.run_pipeline import run_pipeline
from ayn_vqa.stages.asr import Transcript
from ayn_vqa.stages.parse import ParsedTranscript
from ayn_vqa.stages.select import Prediction


class _FakeASRBackend:
    name = "fake-asr"

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript:
        return Transcript(
            record_id, f"transcript {record_id}", self.name, "fake", "msa", 0.01, None
        )


class _FakeParser:
    name = "fake-parse"

    def parse(self, record_id: str, transcript_text: str) -> ParsedTranscript:
        return ParsedTranscript(record_id, "what is it?", ("a", "b", "c"), self.name, None)


class _FakeSelector:
    name = "fake-select"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(
        self, record_id: str, image_path: Path, question: str, options: tuple[str, str, str]
    ) -> Prediction:
        self.calls.append(record_id)
        return Prediction(record_id, 0, None, "fake")


def test_run_pipeline_end_to_end(mini_dataset: Path, tmp_path: Path) -> None:
    selector = _FakeSelector()

    result = run_pipeline(
        data_root=mini_dataset,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "reports",
        experiments_md=tmp_path / "experiments.md",
        split=Split.TRAIN,
        language=Language.MSA,
        asr_backend=_FakeASRBackend(),
        asr_config_key="fake-asr",
        parser=_FakeParser(),
        parser_config_key="fake-parse",
        selector=selector,
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
    )

    assert result.n_records == 7  # all 7 valid train_msa records (see conftest.mini_dataset)
    assert result.n_asr_ok == 7
    assert result.n_parse_ok == 7
    assert result.format_check_ok
    assert result.metrics is not None
    assert len(selector.calls) == 7

    assert result.prediction_csv.exists()
    assert result.error_analysis_csv.exists()

    import pandas as pd

    manifest = pd.read_csv(result.prediction_csv)
    assert len(manifest) == 7
    assert (manifest["prediction"] == 0).all()

    error_table = pd.read_csv(result.error_analysis_csv)
    assert len(error_table) == 7
    assert error_table["question"].eq("what is it?").all()


def test_run_pipeline_reuses_all_stage_caches_on_second_run(
    mini_dataset: Path, tmp_path: Path
) -> None:
    selector = _FakeSelector()
    artifacts_dir = tmp_path / "artifacts"

    for _ in range(2):
        run_pipeline(
            data_root=mini_dataset,
            artifacts_dir=artifacts_dir,
            output_dir=tmp_path / "reports",
            experiments_md=tmp_path / "experiments.md",
            split=Split.TRAIN,
            language=Language.MSA,
            asr_backend=_FakeASRBackend(),
            asr_config_key="fake-asr",
            parser=_FakeParser(),
            parser_config_key="fake-parse",
            selector=selector,
            selector_config_key="fake-select",
            sample_n=3,
            seed=42,
        )

    # same seed -> same 3 sampled records both runs -> selector only ever
    # called for the first run; the second run is served entirely from cache.
    assert len(selector.calls) == 3


def test_run_pipeline_devtest_has_no_labels_and_is_not_scored(
    mini_dataset: Path, tmp_path: Path
) -> None:
    result = run_pipeline(
        data_root=mini_dataset,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "reports",
        experiments_md=tmp_path / "experiments.md",
        split=Split.DEVTEST,
        language=Language.MSA,
        asr_backend=_FakeASRBackend(),
        asr_config_key="fake-asr",
        parser=_FakeParser(),
        parser_config_key="fake-parse",
        selector=_FakeSelector(),
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
    )

    assert result.format_check_ok
    assert result.metrics is None
