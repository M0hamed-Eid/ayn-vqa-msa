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
from ayn_vqa.error_analysis import build_error_table, summarize_by, summarize_option_quality
from ayn_vqa.evaluate import Metric, score_predictions
from ayn_vqa.experiments import log_experiment
from ayn_vqa.logging_utils import setup_logging
from ayn_vqa.option_quality import check_option_quality
from ayn_vqa.stages.asr import ASRBackend, Transcript, WhisperLocalASR
from ayn_vqa.stages.parse import OllamaTranscriptParser, ParsedTranscript, TranscriptParser
from ayn_vqa.stages.retrieve import Exemplar, ExemplarRetriever, OllamaCategoryRetriever
from ayn_vqa.stages.select import Prediction
from ayn_vqa.stages.select_vlm import FALLBACK_INDEX, OllamaJointMCQSelector, VLMSelector
from ayn_vqa.submission import check_submission_format, write_predictions_csv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairSummary:
    """Outcome of the M4 repair-escalation ladder over one pipeline run.
    All zero when repair is disabled or nothing was flagged -- there is
    always a `RepairSummary` on `PipelineResult`, never `None`, so callers
    don't need an extra branch to log it.
    """

    n_flagged: int
    n_resolved_by_reparse: int
    n_resolved_by_reasr: int
    n_still_degenerate: int


@dataclass(frozen=True)
class PipelineResult:
    prediction_csv: Path
    error_analysis_csv: Path
    format_check_ok: bool
    metrics: list[Metric] | None
    n_asr_ok: int
    n_parse_ok: int
    n_records: int
    repair_summary: RepairSummary


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


class _StrictReparseAdapter:
    """Wraps an `OllamaTranscriptParser` to always request its stricter
    repair-retry prompt (`strict=True`), while still exposing the plain
    `parse(record_id, transcript_text) -> ParsedTranscript` shape --
    lets `_run_parse_stage` be reused unmodified for the repair escalation
    ladder's retry passes below, instead of duplicating its caching logic.
    """

    def __init__(self, parser: OllamaTranscriptParser, name: str) -> None:
        self._parser = parser
        self.name = name

    def parse(self, record_id: str, transcript_text: str) -> ParsedTranscript:
        return self._parser.parse(record_id, transcript_text, strict=True)


def _run_repair_stage(
    records: list[Task1aRecord],
    data_root: Path,
    transcripts: dict[str, Transcript],
    parses: dict[str, ParsedTranscript],
    artifacts_dir: Path,
    split: Split,
    language: Language,
    parser: OllamaTranscriptParser,
    parser_config_key: str,
    repair_asr_backend: ASRBackend | None,
    repair_asr_config_key: str,
) -> tuple[dict[str, ParsedTranscript], RepairSummary]:
    """Escalation ladder for parsed items whose options are structurally
    valid but semantically degenerate (`option_quality.check_option_quality`):

    1. Re-parse the *same* transcript with a stricter prompt -- cheap, no
       new ASR call. Fixes the parser-mis-split cases (boilerplate leaking
       into an option, two real options merged into one slot).
    2. If still degenerate, re-transcribe with `repair_asr_backend` (a
       stronger ASR model) and re-parse that. Fixes the cases where the
       transcript itself was insufficient (ASR dropped an option, or
       hallucinated a repetition loop).
    3. If still degenerate, give up -- the original parse is kept and the
       item proceeds to selection exactly as it would have without repair.

    Both steps reuse `_run_parse_stage`/`_run_asr_stage`'s own per-record
    JSONL caching (via distinct config keys), so a second run with the
    same config replays free instead of re-calling any model.
    """
    strict_parser = _StrictReparseAdapter(parser, f"{parser.name}-strict")

    flagged: list[Task1aRecord] = []
    for record in records:
        parsed = parses[record.id]
        if (
            parsed.ok
            and parsed.options is not None
            and check_option_quality(parsed.options).is_degenerate
        ):
            flagged.append(record)
    if not flagged:
        return dict(parses), RepairSummary(0, 0, 0, 0)

    final_parses = dict(parses)

    step1 = _run_parse_stage(
        flagged, transcripts, artifacts_dir, split, language, strict_parser,
        f"{parser_config_key}-repair1",
    )
    n_resolved_by_reparse = 0
    still_flagged: list[Task1aRecord] = []
    for record in flagged:
        candidate = step1[record.id]
        if (
            candidate.ok
            and candidate.options is not None
            and not check_option_quality(candidate.options).is_degenerate
        ):
            final_parses[record.id] = candidate
            n_resolved_by_reparse += 1
            continue
        still_flagged.append(record)

    n_resolved_by_reasr = 0
    if still_flagged and repair_asr_backend is not None:
        repair_transcripts = _run_asr_stage(
            still_flagged, data_root, artifacts_dir, split, language,
            repair_asr_backend, repair_asr_config_key,
        )
        step2 = _run_parse_stage(
            still_flagged, repair_transcripts, artifacts_dir, split, language, strict_parser,
            f"{parser_config_key}-repair2",
        )
        remaining: list[Task1aRecord] = []
        for record in still_flagged:
            candidate = step2[record.id]
            if (
                candidate.ok
                and candidate.options is not None
                and not check_option_quality(candidate.options).is_degenerate
            ):
                final_parses[record.id] = candidate
                n_resolved_by_reasr += 1
                continue
            remaining.append(record)
        still_flagged = remaining

    return final_parses, RepairSummary(
        n_flagged=len(flagged),
        n_resolved_by_reparse=n_resolved_by_reparse,
        n_resolved_by_reasr=n_resolved_by_reasr,
        n_still_degenerate=len(still_flagged),
    )


def _run_retrieval_stage(
    records: list[Task1aRecord],
    retriever: ExemplarRetriever,
    train_records: list[Task1aRecord],
    data_root: Path,
    artifacts_dir: Path,
    language: Language,
    asr_backend: ASRBackend,
    asr_config_key: str,
    parser: TranscriptParser,
    parser_config_key: str,
) -> dict[str, tuple[Exemplar, ...]]:
    """M5 few-shot: which train records to show as worked examples for
    each query, then hydrate each into a full `Exemplar` (question/options
    text + its ground-truth answer).

    Hydration reuses `_run_asr_stage`/`_run_parse_stage` completely
    unmodified, just pointed at `train`'s own cache namespace instead of
    the query split's -- the same trick M4's repair ladder uses to reuse
    those two functions rather than duplicating their caching logic. Only
    the train records actually retrieved by at least one query get
    transcribed/parsed, not all of `train`.
    """
    train_by_id = {record.id: record for record in train_records}

    retrieved_ids: dict[str, tuple[str, ...]] = {}
    all_exemplar_ids: set[str] = set()
    for record in records:
        ids = retriever.retrieve(record.id, record.image_path(data_root))
        retrieved_ids[record.id] = ids
        all_exemplar_ids.update(ids)

    exemplar_records = [train_by_id[eid] for eid in all_exemplar_ids if eid in train_by_id]

    exemplar_transcripts = _run_asr_stage(
        exemplar_records, data_root, artifacts_dir, Split.TRAIN, language,
        asr_backend, asr_config_key,
    )
    exemplar_parses = _run_parse_stage(
        exemplar_records, exemplar_transcripts, artifacts_dir, Split.TRAIN, language,
        parser, parser_config_key,
    )

    hydrated: dict[str, Exemplar] = {}
    for record in exemplar_records:
        parsed = exemplar_parses[record.id]
        if not parsed.ok or parsed.options is None or record.label is None:
            continue  # skip a failed/unusable exemplar rather than show a broken one
        hydrated[record.id] = Exemplar(
            record_id=record.id,
            image_path=record.image_path(data_root),
            question=parsed.question or "",
            options=parsed.options,
            correct_option_text=parsed.options[record.label],
        )

    return {
        record.id: tuple(hydrated[eid] for eid in retrieved_ids[record.id] if eid in hydrated)
        for record in records
    }


def _run_select_stage(
    records: list[Task1aRecord],
    parses: dict[str, ParsedTranscript],
    data_root: Path,
    artifacts_dir: Path,
    split: Split,
    language: Language,
    selector: VLMSelector,
    config_key: str,
    exemplars_by_record: dict[str, tuple[Exemplar, ...]] | None = None,
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
        exemplars = exemplars_by_record.get(record.id, ()) if exemplars_by_record else ()
        prediction = selector.predict(
            record.id,
            record.image_path(data_root),
            parsed.question or "",
            parsed.options,
            exemplars,
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
    repair_enabled: bool = False,
    repair_asr_backend: ASRBackend | None = None,
    repair_asr_config_key: str = "no-repair-asr",
    fewshot_enabled: bool = False,
    fewshot_retriever: ExemplarRetriever | None = None,
    fewshot_train_records: list[Task1aRecord] | None = None,
    fewshot_k: int = 2,
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

    repair_summary = RepairSummary(0, 0, 0, 0)
    effective_selector_config_key = selector_config_key
    if repair_enabled and isinstance(parser, OllamaTranscriptParser):
        # A distinct cache/output key once repair changes what selection
        # sees -- without this, a with-repair and without-repair run would
        # collide on the same select-stage cache and silently serve each
        # other's (wrong-condition) predictions on a second run.
        effective_selector_config_key = f"{selector_config_key}__repair"
        parses, repair_summary = _run_repair_stage(
            records,
            data_root,
            transcripts,
            parses,
            artifacts_dir,
            split,
            language,
            parser,
            parser_config_key,
            repair_asr_backend,
            repair_asr_config_key,
        )
        if repair_summary.n_flagged:
            logger.info(
                "Repair (%s): %d/%d parsed items flagged degenerate -- %d resolved by "
                "reparse, %d resolved by stronger ASR (%s), %d still degenerate",
                parser_config_key,
                repair_summary.n_flagged,
                len(records),
                repair_summary.n_resolved_by_reparse,
                repair_summary.n_resolved_by_reasr,
                repair_asr_config_key if repair_asr_backend is not None else "none configured",
                repair_summary.n_still_degenerate,
            )
    elif repair_enabled:
        logger.warning(
            "repair_enabled=True but parser %r is not an OllamaTranscriptParser -- "
            "skipping repair for this run.",
            parser,
        )

    exemplars_by_record: dict[str, tuple[Exemplar, ...]] | None = None
    if fewshot_enabled and fewshot_retriever is not None and fewshot_train_records is not None:
        # Same reasoning as repair's key suffix above: fewshot changes what
        # selection sees on a per-record basis (exemplars come from a
        # separate retrieval stage, not a property of the selector itself,
        # so it can't just live in `selector.name` the way CoT does), so it
        # needs its own cache/output key too.
        effective_selector_config_key = f"{effective_selector_config_key}__fewshot{fewshot_k}"
        exemplars_by_record = _run_retrieval_stage(
            records,
            fewshot_retriever,
            fewshot_train_records,
            data_root,
            artifacts_dir,
            language,
            asr_backend,
            asr_config_key,
            parser,
            parser_config_key,
        )
        n_with_exemplars = sum(1 for ex in exemplars_by_record.values() if ex)
        logger.info(
            "Retrieval: %d/%d records got at least one exemplar (k=%d requested)",
            n_with_exemplars,
            len(records),
            fewshot_k,
        )

    predictions = _run_select_stage(
        records,
        parses,
        data_root,
        artifacts_dir,
        split,
        language,
        selector,
        effective_selector_config_key,
        exemplars_by_record,
    )
    n_fallback = sum(1 for p in predictions.values() if (p.raw or "").startswith("fallback"))
    n_select_error = sum(1 for p in predictions.values() if (p.raw or "").startswith("error"))
    logger.info(
        "Select (%s): %d predicted, %d fallback (parse failed), %d VLM-call error",
        effective_selector_config_key,
        len(predictions) - n_fallback - n_select_error,
        n_fallback,
        n_select_error,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_key = f"{asr_config_key}__{parser_config_key}__{effective_selector_config_key}"
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
        quality = summarize_option_quality(error_table)
        if quality["n_wrong"]:
            logger.info(
                "Of %d wrong predictions: %d pipeline artifact (degenerate parsed options), "
                "%d genuine visual-reasoning miss",
                quality["n_wrong"],
                quality["n_pipeline_artifact"],
                quality["n_genuine_miss"],
            )

    return PipelineResult(
        prediction_csv=csv_path,
        error_analysis_csv=error_csv_path,
        format_check_ok=check_result.ok,
        metrics=metrics,
        n_asr_ok=n_asr_ok,
        n_parse_ok=n_parse_ok,
        n_records=len(records),
        repair_summary=repair_summary,
    )


def _build_asr_backend(settings: Settings) -> tuple[ASRBackend, str]:
    backend: ASRBackend = WhisperLocalASR(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return backend, f"whisper-{settings.whisper_model_size}"


def _build_repair_asr_backend(settings: Settings) -> tuple[ASRBackend | None, str]:
    """`None` when repair is off -- step 2 of the escalation ladder then
    simply never triggers (flagged items stop at the reparse-only step),
    rather than the caller needing its own branch."""
    if not settings.repair_enabled:
        return None, "no-repair-asr"
    backend: ASRBackend = WhisperLocalASR(
        model_size=settings.repair_whisper_model_size,
        device=settings.repair_whisper_device,
        compute_type=settings.repair_whisper_compute_type,
    )
    return backend, f"whisper-{settings.repair_whisper_model_size}"


def _build_fewshot_retriever(
    settings: Settings, data_root: Path, language: Language
) -> tuple[ExemplarRetriever | None, list[Task1aRecord] | None]:
    """`None, None` when fewshot is off -- mirrors `_build_repair_asr_backend`'s
    shape so `main()` doesn't need its own branch either."""
    if not settings.fewshot_enabled:
        return None, None
    train_split = load_split(data_root, Split.TRAIN, language)
    retriever: ExemplarRetriever = OllamaCategoryRetriever(
        train_split.records,
        k=settings.fewshot_k,
        seed=settings.random_seed,
        model=settings.ollama_select_model,
        base_url=settings.ollama_base_url,
    )
    return retriever, train_split.records


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
    parser.add_argument(
        "--repair-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AYNVQA_REPAIR_ENABLED -- use --no-repair-enabled for an ablation run.",
    )
    parser.add_argument(
        "--cot-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AYNVQA_COT_ENABLED -- use --cot-enabled for an M5 ablation run.",
    )
    parser.add_argument(
        "--fewshot-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AYNVQA_FEWSHOT_ENABLED -- use --fewshot-enabled for an M5 ablation run.",
    )
    parser.add_argument(
        "--fewshot-k", type=int, default=None, help="Override AYNVQA_FEWSHOT_K."
    )
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
    if args.repair_enabled is not None:
        overrides["repair_enabled"] = args.repair_enabled
    if args.cot_enabled is not None:
        overrides["cot_enabled"] = args.cot_enabled
    if args.fewshot_enabled is not None:
        overrides["fewshot_enabled"] = args.fewshot_enabled
    if args.fewshot_k is not None:
        overrides["fewshot_k"] = args.fewshot_k
    if overrides:
        settings = settings.model_copy(update=overrides)

    data_root = args.data_root or settings.resolved_data_root()
    artifacts_dir = args.artifacts_dir or settings.resolved_artifacts_dir()
    output_dir = args.output_dir or (settings.resolved_reports_dir() / "m3_pipeline")
    sample_n = args.sample_n if args.sample_n is not None else settings.pipeline_sample_n
    seed = args.seed if args.seed is not None else settings.random_seed

    asr_backend, asr_config_key = _build_asr_backend(settings)
    repair_asr_backend, repair_asr_config_key = _build_repair_asr_backend(settings)
    fewshot_retriever, fewshot_train_records = _build_fewshot_retriever(
        settings, data_root, Language(args.language)
    )
    parser_stage = OllamaTranscriptParser(
        model=settings.ollama_parse_model, base_url=settings.ollama_base_url
    )
    selector: VLMSelector = OllamaJointMCQSelector(
        model=settings.ollama_select_model,
        base_url=settings.ollama_base_url,
        use_cot=settings.cot_enabled,
        fewshot_num_ctx=settings.fewshot_num_ctx,
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
        repair_enabled=settings.repair_enabled,
        repair_asr_backend=repair_asr_backend,
        repair_asr_config_key=repair_asr_config_key,
        fewshot_enabled=settings.fewshot_enabled,
        fewshot_retriever=fewshot_retriever,
        fewshot_train_records=fewshot_train_records,
        fewshot_k=settings.fewshot_k,
    )

    logger.info("Prediction CSV: %s", result.prediction_csv)
    logger.info("Error analysis: %s", result.error_analysis_csv)
    if not result.format_check_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
