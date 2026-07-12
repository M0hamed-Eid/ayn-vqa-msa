"""End-to-end test of the M3 cascade: ASR -> parse -> select -> submit ->
score -> error analysis, against the synthetic `mini_dataset` fixture,
using fake stages (no real Whisper, no real Ollama) so it runs instantly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pandas as pd

from ayn_vqa.data.loader import load_split
from ayn_vqa.data.schema import Language, Split
from ayn_vqa.run_pipeline import RepairSummary, run_pipeline
from ayn_vqa.stages.asr import ASRBackend, Transcript
from ayn_vqa.stages.parse import OllamaTranscriptParser, ParsedTranscript
from ayn_vqa.stages.retrieve import Exemplar
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
        self,
        record_id: str,
        image_path: Path,
        question: str,
        options: tuple[str, str, str],
        exemplars: tuple[Exemplar, ...] = (),
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


class _FakeRetriever:
    name = "fake-retriever"

    def __init__(self, exemplar_ids: tuple[str, ...]) -> None:
        self._exemplar_ids = exemplar_ids

    def retrieve(self, record_id: str, image_path: Path) -> tuple[str, ...]:
        return self._exemplar_ids


class _ExemplarCapturingSelector:
    name = "fake-select"

    def __init__(self) -> None:
        self.seen_exemplars: list[tuple[Exemplar, ...]] = []

    def predict(
        self,
        record_id: str,
        image_path: Path,
        question: str,
        options: tuple[str, str, str],
        exemplars: tuple[Exemplar, ...] = (),
    ) -> Prediction:
        self.seen_exemplars.append(exemplars)
        return Prediction(record_id, 0, None, "fake")


def test_run_pipeline_fewshot_hydrates_and_passes_exemplars(
    mini_dataset: Path, tmp_path: Path
) -> None:
    train_records = load_split(mini_dataset, Split.TRAIN, Language.MSA).records
    retriever = _FakeRetriever(exemplar_ids=(train_records[0].id, train_records[1].id))
    selector = _ExemplarCapturingSelector()

    result = run_pipeline(
        data_root=mini_dataset,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "reports",
        experiments_md=tmp_path / "experiments.md",
        split=Split.DEV,
        language=Language.MSA,
        asr_backend=_FakeASRBackend(),
        asr_config_key="fake-asr",
        parser=_FakeParser(),
        parser_config_key="fake-parse",
        selector=selector,
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
        fewshot_enabled=True,
        fewshot_retriever=retriever,
        fewshot_train_records=train_records,
        fewshot_k=2,
    )

    assert len(selector.seen_exemplars) == 1  # dev fixture is a single row
    exemplars = selector.seen_exemplars[0]
    assert len(exemplars) == 2
    assert {e.record_id for e in exemplars} == {train_records[0].id, train_records[1].id}
    assert train_records[0].label is not None
    assert exemplars[0].correct_option_text == exemplars[0].options[train_records[0].label]
    # fewshot changes what select sees -- distinct cache/output key, same
    # reasoning as repair's `__repair` suffix.
    assert "__fewshot2" in result.prediction_csv.name


_DEGENERATE_PAYLOAD = {
    "question": "q",
    "option_0": "خيار أول",
    "option_1": "خيار ثاني",
    "option_2": "الخيارات هي",
}
_CLEAN_PAYLOAD = {
    "question": "q",
    "option_0": "خيار أول",
    "option_1": "خيار ثاني",
    "option_2": "خيار ثالث",
}


class _RepairASRBackend:
    """Stands in for a stronger ASR model (e.g. whisper-large-v3): returns
    a transcript distinguishable from `_FakeASRBackend`'s, so tests can
    tell step 2 of the repair ladder apart from step 1."""

    name = "fake-repair-asr"

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript:
        return Transcript(
            record_id, f"REPAIRED-transcript {record_id}", self.name, "fake", "msa", 0.01, None
        )


def _scripted_ollama_parser(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaTranscriptParser:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaTranscriptParser(client=client)


def test_run_pipeline_repair_resolves_via_reparse(mini_dataset: Path, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.read())["messages"][0]["content"]
        payload = _CLEAN_PAYLOAD if "previous attempt" in prompt else _DEGENERATE_PAYLOAD
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

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
        parser=_scripted_ollama_parser(handler),
        parser_config_key="ollama-parse",
        selector=selector,
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
        repair_enabled=True,
    )

    assert result.repair_summary == RepairSummary(
        n_flagged=7, n_resolved_by_reparse=7, n_resolved_by_reasr=0, n_still_degenerate=0
    )
    error_table = pd.read_csv(result.error_analysis_csv)
    assert error_table["option_quality_ok"].all()
    assert len(selector.calls) == 7  # every record still reached selection


def test_run_pipeline_repair_resolves_via_asr_escalation(
    mini_dataset: Path, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.read())["messages"][0]["content"]
        resolved = "previous attempt" in prompt and "REPAIRED-transcript" in prompt
        payload = _CLEAN_PAYLOAD if resolved else _DEGENERATE_PAYLOAD
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

    repair_backend: ASRBackend = _RepairASRBackend()
    result = run_pipeline(
        data_root=mini_dataset,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "reports",
        experiments_md=tmp_path / "experiments.md",
        split=Split.TRAIN,
        language=Language.MSA,
        asr_backend=_FakeASRBackend(),
        asr_config_key="fake-asr",
        parser=_scripted_ollama_parser(handler),
        parser_config_key="ollama-parse",
        selector=_FakeSelector(),
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
        repair_enabled=True,
        repair_asr_backend=repair_backend,
        repair_asr_config_key="fake-repair-asr",
    )

    assert result.repair_summary == RepairSummary(
        n_flagged=7, n_resolved_by_reparse=0, n_resolved_by_reasr=7, n_still_degenerate=0
    )


def test_run_pipeline_repair_gives_up_gracefully_when_unresolved(
    mini_dataset: Path, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps(_DEGENERATE_PAYLOAD)}})

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
        parser=_scripted_ollama_parser(handler),
        parser_config_key="ollama-parse",
        selector=selector,
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
        repair_enabled=True,
    )

    assert result.repair_summary == RepairSummary(
        n_flagged=7, n_resolved_by_reparse=0, n_resolved_by_reasr=0, n_still_degenerate=7
    )
    # never silently drops a record -- it proceeds to selection best-effort
    assert len(selector.calls) == 7


def test_run_pipeline_repair_is_a_no_op_when_nothing_flagged(
    mini_dataset: Path, tmp_path: Path
) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"message": {"content": json.dumps(_CLEAN_PAYLOAD)}})

    result = run_pipeline(
        data_root=mini_dataset,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "reports",
        experiments_md=tmp_path / "experiments.md",
        split=Split.TRAIN,
        language=Language.MSA,
        asr_backend=_FakeASRBackend(),
        asr_config_key="fake-asr",
        parser=_scripted_ollama_parser(handler),
        parser_config_key="ollama-parse",
        selector=_FakeSelector(),
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
        repair_enabled=True,
    )

    assert result.repair_summary == RepairSummary(0, 0, 0, 0)
    assert call_count == 7  # one parse call per record -- no retries triggered


def test_run_pipeline_repair_skips_gracefully_for_non_ollama_parser(
    mini_dataset: Path, tmp_path: Path
) -> None:
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
        selector=_FakeSelector(),
        selector_config_key="fake-select",
        sample_n=None,
        seed=42,
        repair_enabled=True,
    )

    assert result.repair_summary == RepairSummary(0, 0, 0, 0)
    assert result.n_records == 7


def test_run_pipeline_repair_on_and_off_use_separate_caches_and_outputs(
    mini_dataset: Path, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.read())["messages"][0]["content"]
        payload = _CLEAN_PAYLOAD if "previous attempt" in prompt else _DEGENERATE_PAYLOAD
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

    artifacts_dir = tmp_path / "artifacts"

    def _run(repair_enabled: bool) -> pd.DataFrame:
        result = run_pipeline(
            data_root=mini_dataset,
            artifacts_dir=artifacts_dir,
            output_dir=tmp_path / "reports",
            experiments_md=tmp_path / "experiments.md",
            split=Split.TRAIN,
            language=Language.MSA,
            asr_backend=_FakeASRBackend(),
            asr_config_key="fake-asr",
            parser=_scripted_ollama_parser(handler),
            parser_config_key="ollama-parse",
            selector=_FakeSelector(),
            selector_config_key="fake-select",
            sample_n=None,
            seed=42,
            repair_enabled=repair_enabled,
        )
        return pd.read_csv(result.error_analysis_csv)

    off_table = _run(False)
    on_table = _run(True)

    # repair off never touched the strict-retry path -> stayed degenerate;
    # repair on resolved it -- if the two runs shared a cache, one of
    # these would incorrectly show the other condition's result.
    assert not off_table["option_quality_ok"].any()
    assert on_table["option_quality_ok"].all()
