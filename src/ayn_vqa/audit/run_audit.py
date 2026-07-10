"""CLI entry point: `uv run aynvqa-audit` (or `python -m ayn_vqa.audit.run_audit`).

Orchestrates every other module in this package into one M0 data-audit run:
load JSONL -> validate media paths -> probe audio/image stats -> hash images
for duplicates -> render a random sample grid -> write CSV/JSON/Markdown
reports. Each step is a plain function imported from another module; this
file only wires them together, parses CLI args, and logs a console summary.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ayn_vqa.audit.audio_stats import probe_audio
from ayn_vqa.audit.hashing import dhash, find_near_duplicates, group_exact_duplicates, sha256_file
from ayn_vqa.audit.image_stats import probe_image
from ayn_vqa.audit.report import ReportPaths, SplitAuditData, write_reports
from ayn_vqa.audit.sampling import render_sample_grid, sample_records
from ayn_vqa.config import get_settings
from ayn_vqa.data.loader import load_split
from ayn_vqa.data.schema import Language, Split, Task1aRecord
from ayn_vqa.data.validation import missing_media, validate_split_media
from ayn_vqa.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_ALL_SPLITS = (Split.TRAIN, Split.DEV, Split.DEVTEST)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0 data audit for Ayn-VQA Task 1a.")
    parser.add_argument(
        "--language",
        choices=[lang.value for lang in Language],
        default=Language.MSA.value,
        help="Audio track to audit (default: msa -- the competition track for this project).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=[split.value for split in Split],
        default=[split.value for split in _ALL_SPLITS],
        help="Splits to audit (default: train dev devtest).",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override AYNVQA_DATA_ROOT.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override the report output directory."
    )
    parser.add_argument("--sample-n", type=int, default=None, help="Override the sample grid size.")
    parser.add_argument(
        "--near-dup-max-distance",
        type=int,
        default=None,
        help="Override the near-duplicate perceptual-hash Hamming threshold.",
    )
    parser.add_argument(
        "--log-level", default=None, help="Override the log level (DEBUG/INFO/WARNING)."
    )
    return parser.parse_args(argv)


def _hash_images(
    records: list[Task1aRecord], data_root: Path
) -> tuple[dict[str, str], dict[str, int]]:
    sha_by_id: dict[str, str] = {}
    phash_by_id: dict[str, int] = {}
    for record in records:
        path = record.image_path(data_root)
        if not path.is_file():
            continue  # already reported as missing by validate_split_media
        try:
            sha_by_id[record.id] = sha256_file(path)
            phash_by_id[record.id] = dhash(path)
        except Exception:
            logger.exception("Failed to hash image for record %s (%s)", record.id, path)
    return sha_by_id, phash_by_id


def run_audit(
    data_root: Path,
    output_dir: Path,
    language: Language,
    splits: tuple[Split, ...],
    sample_n: int,
    near_dup_max_distance: int,
    seed: int,
) -> ReportPaths:
    bundles: list[SplitAuditData] = []

    for split in splits:
        logger.info("Loading %s/%s ...", split.value, language.value)
        split_data = load_split(data_root, split, language)
        logger.info(
            "%s/%s: %d records loaded, %d parse errors",
            split.value,
            language.value,
            len(split_data),
            len(split_data.errors),
        )

        media_checks = validate_split_media(split_data, data_root)
        n_missing = len(missing_media(media_checks))
        if n_missing:
            logger.warning("%s/%s: %d missing media files", split.value, language.value, n_missing)

        audio_stats = [probe_audio(r.id, r.audio_path(data_root)) for r in split_data.records]
        image_stats = [probe_image(r.id, r.image_path(data_root)) for r in split_data.records]

        bundles.append(
            SplitAuditData(
                split_data=split_data,
                media_checks=media_checks,
                audio_stats=audio_stats,
                image_stats=image_stats,
            )
        )

    logger.info("Hashing images for duplicate detection ...")
    all_records = [r for bundle in bundles for r in bundle.split_data.records]
    sha_by_id, phash_by_id = _hash_images(all_records, data_root)

    exact_groups = group_exact_duplicates(sha_by_id)
    near_pairs = find_near_duplicates(phash_by_id, max_distance=near_dup_max_distance)
    logger.info(
        "Duplicate scan: %d exact-duplicate groups, %d near-duplicate pairs",
        len(exact_groups),
        len(near_pairs),
    )

    logger.info("Rendering random sample grid (n=%d, seed=%d) ...", sample_n, seed)
    sampled = sample_records(all_records, sample_n, seed)
    sample_grid_path = render_sample_grid(
        sampled, data_root, output_dir / "figures" / "sample_grid.png"
    )

    return write_reports(bundles, exact_groups, near_pairs, output_dir, sample_grid_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()

    data_root = args.data_root or settings.resolved_data_root()
    output_dir = args.output_dir or (settings.resolved_reports_dir() / "m0_data_audit")
    sample_n = args.sample_n or settings.sample_grid_n
    near_dup_max_distance = (
        args.near_dup_max_distance
        if args.near_dup_max_distance is not None
        else settings.near_dup_max_distance
    )
    log_level = args.log_level or settings.log_level

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level, log_file=output_dir / "audit.log")

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", output_dir)

    start = time.monotonic()
    report_paths = run_audit(
        data_root=data_root,
        output_dir=output_dir,
        language=Language(args.language),
        splits=tuple(Split(s) for s in args.splits),
        sample_n=sample_n,
        near_dup_max_distance=near_dup_max_distance,
        seed=settings.random_seed,
    )
    elapsed = time.monotonic() - start

    logger.info("Audit complete in %.1fs.", elapsed)
    logger.info("Report: %s", report_paths.report_md)
    logger.info("Manifest: %s", report_paths.manifest_csv)
    logger.info("Summary JSON: %s", report_paths.summary_json)


if __name__ == "__main__":
    main()
