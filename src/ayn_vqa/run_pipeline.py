"""CLI entry point: `uv run aynvqa-run`.

The M3 cascade: load a split -> transcribe (reusing M2's cache if
present, else running Whisper fresh) -> parse each transcript into
question+options -> ask the VLM to pick an option, jointly, for each ->
write `prediction.csv` -> format-check -> (if labeled) score -> error
analysis report. Every stage's output is cached under
`artifacts/<split>/<stage>/` (`ayn_vqa.artifacts`, from M2), so re-running
after a crash or a config tweak only recomputes what actually changed --
and because M2's ASR bench already cached `whisper-medium` transcripts for
`dev`'s seeded 50-item sample, a same-config first run of this pipeline
costs zero new ASR calls.

Producing `prediction.csv` is as far as this goes automatically: actually
uploading it to Codabench is a manual step left to you (see the README) --
that's a submission with real, limited-attempts consequences, not
something to automate away.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from ayn_vqa.artifacts import append_jsonl, artifact_path, read_jsonl_cache
from ayn_vqa.audit.sampling import sample_records
from ayn_vqa.config import PROJECT_ROOT, Settings, get_settings
from ayn_vqa.data.loader import load_split
from ayn_vqa.data.schema import Language, Split, Task1aRecord
from ayn_vqa.error_analysis import build_error_table, summarize_by
from ayn_vqa.evaluate import Metric, score_predictions
from ayn_vqa.experiments import log_experiment
from ayn_vqa.logging_utils import setup_logging
from ayn_vqa.stages.asr import ASRBackend, Transcript, WhisperLocalASR
from ayn_vqa.stages.parse import OllamaTranscriptParser, ParsedTranscript, TranscriptParser
from ayn_vqa.stages.select import Prediction
from ayn_vqa.stages.select_vlm import FALLBACK_INDEX, OllamaJointMCQSelector, VLMSelector
from ayn_vqa.submission import check_submission_format, write_predictions_csv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    prediction_csv: Path
    error_analysis_csv: Path
    format_check_ok: bool
    metrics: list[Metric] | None
    n_asr_ok: int
    n_parse_ok: int
    n_records: int


def _run_asr_stage(
    records: list[Task1aRecord],
    data_root: Path,
    artifacts_dir: Path,
    split: Split,
    language: Language,
    backend: ASRBackend,
    config_key: str,
) -> dict[str, Transcript]:
    cache_path = artifact_path(artifacts_dir, split.value, "asr", f"{language.value}_{config_key}")
    cache = read_jsonl_cache(cache_path)
    transcripts: dict[str, Transcript] = {}
    for record in records:
        if record.id in cache:
            transcripts[record.id] = Transcript(**cache[record.id])
            continue
        transcript = backend.transcribe(record.id, record.audio_path(data_root))
        append_jsonl(cache_path, asdict(transcript))
        transcripts[record.id] = transcript
    return transcripts


def _run_parse_stage(
    records: list[Task1aRecord],
    transcripts: dict[str, Transcript],
    artifacts_dir: Path,
    split: Split,
    language: Language,
    parser: TranscriptParser,
    config_key: str,
) -> dict[str, ParsedTranscript]:
    cache_path = artifact_path(
        artifacts_dir, split.value, "parse", f"{language.value}_{config_key}"
    )
    cache = read_jsonl_cache(cache_path)
    parses: dict[str, ParsedTranscript] = {}
    for record in records:
        transcript = transcripts[record.id]
        if not transcript.ok or not transcript.text:
            parses[record.id] = ParsedTranscript(
                record.id, None, None, parser.name, transcript.error or "empty transcript"
            )
            continue
        if record.id in cache:
            row = dict(cache[record.id])
            if row.get("options") is not None:
                row["options"] = tuple(row["options"])
            parses[record.id] = ParsedTranscript(**row)
            continue
        parsed = parser.parse(record.id, transcript.text)
        append_jsonl(cache_path, asdict(parsed))
        parses[record.id] = parsed
    return parses


def _run_select_stage(
    records: list[Task1aRecord],
    parses: dict[str, ParsedTranscript],
    data_root: Path,
    artifacts_dir: Path,
    split: Split,
    language: Language,
    selector: VLMSelector,
    config_key: str,
) -> dict[str, Prediction]:
    cache_path = artifact_path(
        artifacts_dir, split.value, "select", f"{language.value}_{config_key}"
    )
    cache = read_jsonl_cache(cache_path)
    predictions: dict[str, Prediction] = {}
    for record in records:
        parsed = parses[record.id]
        if not parsed.ok or parsed.options is None:
            predictions[record.id] = Prediction(
                record.id, FALLBACK_INDEX, None, f"fallback: parse failed ({parsed.error})"
            )
            continue
        if record.id in cache:
            predictions[record.id] = Prediction(**cache[record.id])
            continue
        prediction = selector.predict(
            record.id, record.image_path(data_root), parsed.question or "", parsed.options
        )
        append_jsonl(cache_path, asdict(prediction))
        predictions[record.id] = prediction
    return predictions


def run_pipeline(
    data_root: Path,
    artifacts_dir: Path,
    output_dir: Path,
    experiments_md: Path,
    split: Split,
    language: Language,
    asr_backend: ASRBackend,
    asr_config_key: str,
    parser: TranscriptParser,
    parser_config_key: str,
    selector: VLMSelector,
    selector_config_key: str,
    sample_n: int | None,
    seed: int,
) -> PipelineResult:
    split_data = load_split(data_root, split, language)
    records = (
        sample_records(split_data.records, sample_n, seed)
        if sample_n is not None
        else split_data.records
    )
    logger.info(
        "%s/%s: running pipeline over %d/%d records",
        split.value,
        language.value,
        len(records),
        len(split_data),
    )

    transcripts = _run_asr_stage(
        records, data_root, artifacts_dir, split, language, asr_backend, asr_config_key
    )
    n_asr_ok = sum(1 for t in transcripts.values() if t.ok)
    logger.info("ASR (%s): %d/%d succeeded", asr_config_key, n_asr_ok, len(records))

    parses = _run_parse_stage(
        records, transcripts, artifacts_dir, split, language, parser, parser_config_key
    )
    n_parse_ok = sum(1 for p in parses.values() if p.ok)
    logger.info("Parse (%s): %d/%d succeeded", parser_config_key, n_parse_ok, len(records))

    predictions = _run_select_stage(
        records, parses, data_root, artifacts_dir, split, language, selector, selector_config_key
    )
    n_fallback = sum(1 for p in predictions.values() if (p.raw or "").startswith("fallback"))
    n_select_error = sum(1 for p in predictions.values() if (p.raw or "").startswith("error"))
    logger.info(
        "Select (%s): %d predicted, %d fallback (parse failed), %d VLM-call error",
        selector_config_key,
        len(predictions) - n_fallback - n_select_error,
        n_fallback,
        n_select_error,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_key = f"{asr_config_key}__{parser_config_key}__{selector_config_key}"
    csv_path = output_dir / f"prediction_{config_key}_{split.value}_{language.value}.csv"
    write_predictions_csv(list(predictions.values()), csv_path)
    logger.info("Wrote %d predictions to %s", len(predictions), csv_path)

    gold_ids = [record.id for record in records]
    check_result = check_submission_format(csv_path, gold_ids=gold_ids)
    for warning in check_result.warnings:
        logger.warning("format checker: %s", warning)
    for error in check_result.errors:
        logger.error("format checker: %s", error)
    logger.info("Format check: %s", "PASSED" if check_result.ok else "FAILED")

    metrics: list[Metric] | None = None
    has_labels = any(record.label is not None for record in records)
    if has_labels:
        metrics = score_predictions(records, csv_path)
        for metric in metrics:
            marker = "  <-- RANKING" if metric.is_ranking else ""
            logger.info("%-18s %.4f%s", metric.name, metric.value, marker)
        log_experiment(experiments_md, config_key, split.value, language.value, metrics)
    else:
        logger.info(
            "%s/%s has no labels -- skipping scoring (devtest/test are unlabeled).",
            split.value,
            language.value,
        )

    error_table = build_error_table(
        records,
        {record_id: asdict(t) for record_id, t in transcripts.items()},
        {record_id: asdict(p) for record_id, p in parses.items()},
        {record_id: p.pred for record_id, p in predictions.items()},
    )
    error_csv_path = output_dir / f"error_analysis_{config_key}_{split.value}_{language.value}.csv"
    error_table.to_csv(error_csv_path, index=False)
    logger.info("Wrote error analysis to %s", error_csv_path)

    if has_labels:
        for column in ("country", "category"):
            breakdown = summarize_by(error_table, column)
            if not breakdown.empty:
                logger.info(
                    "Accuracy by %s (worst first):\n%s", column, breakdown.to_string(index=False)
                )

    return PipelineResult(
        prediction_csv=csv_path,
        error_analysis_csv=error_csv_path,
        format_check_ok=check_result.ok,
        metrics=metrics,
        n_asr_ok=n_asr_ok,
        n_parse_ok=n_parse_ok,
        n_records=len(records),
    )


def _build_asr_backend(settings: Settings) -> tuple[ASRBackend, str]:
    backend: ASRBackend = WhisperLocalASR(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return backend, f"whisper-{settings.whisper_model_size}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M3 cascade: ASR -> parse -> joint-MCQ VLM -> submit -> score -> errors."
    )
    parser.add_argument("--split", choices=[s.value for s in Split], default=Split.DEV.value)
    parser.add_argument(
        "--language", choices=[lang.value for lang in Language], default=Language.MSA.value
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Run only a seeded sample of this size (default: the whole split).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override AYNVQA_RANDOM_SEED.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--whisper-model-size", default=None, help="Override AYNVQA_WHISPER_MODEL_SIZE."
    )
    parser.add_argument(
        "--ollama-parse-model", default=None, help="Override AYNVQA_OLLAMA_PARSE_MODEL."
    )
    parser.add_argument(
        "--ollama-select-model", default=None, help="Override AYNVQA_OLLAMA_SELECT_MODEL."
    )
    parser.add_argument("--ollama-base-url", default=None, help="Override AYNVQA_OLLAMA_BASE_URL.")
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(args.log_level or settings.log_level)

    overrides: dict[str, object] = {}
    if args.whisper_model_size:
        overrides["whisper_model_size"] = args.whisper_model_size
    if args.ollama_parse_model:
        overrides["ollama_parse_model"] = args.ollama_parse_model
    if args.ollama_select_model:
        overrides["ollama_select_model"] = args.ollama_select_model
    if args.ollama_base_url:
        overrides["ollama_base_url"] = args.ollama_base_url
    if overrides:
        settings = settings.model_copy(update=overrides)

    data_root = args.data_root or settings.resolved_data_root()
    artifacts_dir = args.artifacts_dir or settings.resolved_artifacts_dir()
    output_dir = args.output_dir or (settings.resolved_reports_dir() / "m3_pipeline")
    sample_n = args.sample_n if args.sample_n is not None else settings.pipeline_sample_n
    seed = args.seed if args.seed is not None else settings.random_seed

    asr_backend, asr_config_key = _build_asr_backend(settings)
    parser_stage: TranscriptParser = OllamaTranscriptParser(
        model=settings.ollama_parse_model, base_url=settings.ollama_base_url
    )
    selector: VLMSelector = OllamaJointMCQSelector(
        model=settings.ollama_select_model, base_url=settings.ollama_base_url
    )

    result = run_pipeline(
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        experiments_md=PROJECT_ROOT / "experiments.md",
        split=Split(args.split),
        language=Language(args.language),
        asr_backend=asr_backend,
        asr_config_key=asr_config_key,
        parser=parser_stage,
        parser_config_key=parser_stage.name,
        selector=selector,
        selector_config_key=selector.name,
        sample_n=sample_n,
        seed=seed,
    )

    logger.info("Prediction CSV: %s", result.prediction_csv)
    logger.info("Error analysis: %s", result.error_analysis_csv)
    if not result.format_check_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
