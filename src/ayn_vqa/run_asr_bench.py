"""CLI entry point: `uv run aynvqa-transcribe`.

Runs one ASR backend over a seeded sample of one split, caching every
transcript to `artifacts/<split>/asr/<language>_<config_key>.jsonl` (so
re-running the same config is instant) and printing a console summary:
success/failure counts, latency stats, transcript length stats, and a
handful of full transcripts to spot-check by ear. This process can measure
latency and length; only a native speaker can judge whether the Arabic
transcript is actually right -- that judgment call is deliberately left
to the console output, not automated away.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from ayn_vqa.artifacts import append_jsonl, artifact_path, read_jsonl_cache
from ayn_vqa.audit.sampling import sample_records
from ayn_vqa.config import Settings, get_settings
from ayn_vqa.data.loader import load_split
from ayn_vqa.data.schema import Language, Split
from ayn_vqa.logging_utils import setup_logging
from ayn_vqa.stages.asr import (
    ASRBackend,
    FanarAuraASR,
    OpenAITranscribeASR,
    Transcript,
    WhisperLocalASR,
)

logger = logging.getLogger(__name__)


def _build_backend(name: str, settings: Settings) -> tuple[ASRBackend, str]:
    if name == "whisper":
        backend: ASRBackend = WhisperLocalASR(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        return backend, f"whisper-{settings.whisper_model_size}"
    if name == "fanar":
        backend = FanarAuraASR(
            api_key=settings.fanar_api_key,
            model=settings.fanar_stt_model,
            base_url=settings.fanar_base_url,
        )
        return backend, "fanar-aura-stt"
    if name == "openai":
        backend = OpenAITranscribeASR(
            api_key=settings.openai_api_key,
            model=settings.openai_transcribe_model,
            base_url=settings.openai_base_url,
        )
        return backend, f"openai-{settings.openai_transcribe_model}"
    raise ValueError(f"unknown backend {name!r}")


def run_asr_bench(
    data_root: Path,
    artifacts_dir: Path,
    split: Split,
    language: Language,
    config_key: str,
    backend: ASRBackend,
    sample_n: int,
    seed: int,
) -> Path:
    split_data = load_split(data_root, split, language)
    sampled = sample_records(split_data.records, sample_n, seed)
    logger.info(
        "%s/%s: sampled %d/%d records (seed=%d)",
        split.value,
        language.value,
        len(sampled),
        len(split_data),
        seed,
    )

    cache_path = artifact_path(artifacts_dir, split.value, "asr", f"{language.value}_{config_key}")
    cached = read_jsonl_cache(cache_path)
    logger.info("Cache: %s (%d already cached)", cache_path, len(cached))

    transcripts: list[Transcript] = []
    n_new = 0
    for record in sampled:
        if record.id in cached:
            transcripts.append(Transcript(**cached[record.id]))
            continue
        transcript = backend.transcribe(record.id, record.audio_path(data_root))
        append_jsonl(cache_path, asdict(transcript))
        transcripts.append(transcript)
        n_new += 1
        if not transcript.ok:
            logger.warning("%s: %s", record.id, transcript.error)

    logger.info("Transcribed %d new, %d from cache", n_new, len(sampled) - n_new)

    ok = [t for t in transcripts if t.ok]
    logger.info("%d/%d succeeded", len(ok), len(transcripts))
    if ok:
        lengths = [len(t.text or "") for t in ok]
        latencies = [t.latency_sec for t in ok if t.latency_sec is not None]
        logger.info(
            "Transcript length (chars): mean=%.0f min=%d max=%d",
            sum(lengths) / len(lengths),
            min(lengths),
            max(lengths),
        )
        if latencies:
            logger.info(
                "Latency (s): mean=%.2f min=%.2f max=%.2f",
                sum(latencies) / len(latencies),
                min(latencies),
                max(latencies),
            )
        logger.info("Sample transcripts (spot-check by ear against the audio):")
        for transcript in ok[:5]:
            logger.info("  [%s] %s", transcript.record_id[:10], transcript.text)

    return cache_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2 ASR bench: transcribe a seeded sample of one split with one backend."
    )
    parser.add_argument("--backend", choices=["whisper", "fanar", "openai"], default="whisper")
    parser.add_argument("--split", choices=[s.value for s in Split], default=Split.DEV.value)
    parser.add_argument(
        "--language", choices=[lang.value for lang in Language], default=Language.MSA.value
    )
    parser.add_argument("--sample-n", type=int, default=None, help="Override AYNVQA_ASR_SAMPLE_N.")
    parser.add_argument("--seed", type=int, default=None, help="Override AYNVQA_RANDOM_SEED.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument(
        "--whisper-model-size",
        default=None,
        help="Override AYNVQA_WHISPER_MODEL_SIZE (only used with --backend whisper).",
    )
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(args.log_level or settings.log_level)

    if args.whisper_model_size:
        settings = settings.model_copy(update={"whisper_model_size": args.whisper_model_size})

    data_root = args.data_root or settings.resolved_data_root()
    artifacts_dir = args.artifacts_dir or settings.resolved_artifacts_dir()
    sample_n = args.sample_n or settings.asr_sample_n
    seed = args.seed if args.seed is not None else settings.random_seed

    backend, config_key = _build_backend(args.backend, settings)

    cache_path = run_asr_bench(
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        split=Split(args.split),
        language=Language(args.language),
        config_key=config_key,
        backend=backend,
        sample_n=sample_n,
        seed=seed,
    )
    logger.info("Cached transcripts: %s", cache_path)


if __name__ == "__main__":
    main()
