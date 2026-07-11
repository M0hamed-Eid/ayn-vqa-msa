"""CLI entry point: `uv run aynvqa-predict`.

Runs a trivial baseline selector (random or constant) over one split,
writes `prediction.csv`, checks its format against the vendored checker,
and -- if the split has gold labels -- scores it with the vendored scorer
and appends a row to `experiments.md`. No ASR/VLM stage exists yet; these
two baselines exist purely to prove the load -> predict -> submit -> score
loop works end to end before a real model is ever plugged into it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from ayn_vqa.config import PROJECT_ROOT, get_settings
from ayn_vqa.data.loader import load_split
from ayn_vqa.data.schema import Language, Split
from ayn_vqa.evaluate import Metric, score_predictions
from ayn_vqa.experiments import log_experiment
from ayn_vqa.logging_utils import setup_logging
from ayn_vqa.stages.select import AnswerSelector, ConstantSelector, RandomSelector, predict_split
from ayn_vqa.submission import check_submission_format, write_predictions_csv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineResult:
    prediction_csv: Path
    format_check_ok: bool
    metrics: list[Metric] | None


def _build_selector(name: str, constant_value: int, seed: int) -> tuple[AnswerSelector, str]:
    if name == "random":
        return RandomSelector(seed=seed), "random"
    return ConstantSelector(value=constant_value), f"constant-{constant_value}"


def run_baseline(
    data_root: Path,
    output_dir: Path,
    experiments_md: Path,
    selector_name: str,
    constant_value: int,
    split: Split,
    language: Language,
    seed: int,
) -> BaselineResult:
    split_data = load_split(data_root, split, language)
    logger.info("%s/%s: %d records loaded", split.value, language.value, len(split_data))

    selector, resolved_name = _build_selector(selector_name, constant_value, seed)
    predictions = predict_split(selector, split_data.records)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"prediction_{resolved_name}_{split.value}_{language.value}.csv"
    write_predictions_csv(predictions, csv_path)
    logger.info("Wrote %d predictions to %s", len(predictions), csv_path)

    gold_ids = [record.id for record in split_data.records]
    check_result = check_submission_format(csv_path, gold_ids=gold_ids)
    for warning in check_result.warnings:
        logger.warning("format checker: %s", warning)
    for error in check_result.errors:
        logger.error("format checker: %s", error)
    logger.info("Format check: %s", "PASSED" if check_result.ok else "FAILED")

    metrics: list[Metric] | None = None
    has_labels = any(record.label is not None for record in split_data.records)
    if has_labels:
        metrics = score_predictions(split_data.records, csv_path)
        for metric in metrics:
            marker = "  <-- RANKING" if metric.is_ranking else ""
            logger.info("%-18s %.4f%s", metric.name, metric.value, marker)
        log_experiment(experiments_md, resolved_name, split.value, language.value, metrics)
    else:
        logger.info(
            "%s/%s has no labels -- skipping scoring (devtest/test are unlabeled).",
            split.value,
            language.value,
        )

    return BaselineResult(prediction_csv=csv_path, format_check_ok=check_result.ok, metrics=metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M1 trivial-baseline harness: load -> predict -> submit -> score."
    )
    parser.add_argument("--selector", choices=["random", "constant"], default="random")
    parser.add_argument(
        "--constant-value",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Only used by --selector constant.",
    )
    parser.add_argument(
        "--split",
        choices=[s.value for s in Split],
        default=Split.DEV.value,
        help="train/dev are scored against gold labels; devtest is not.",
    )
    parser.add_argument(
        "--language", choices=[lang.value for lang in Language], default=Language.MSA.value
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override AYNVQA_DATA_ROOT.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override the prediction output directory."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override AYNVQA_RANDOM_SEED.")
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(args.log_level or settings.log_level)

    data_root = args.data_root or settings.resolved_data_root()
    output_dir = args.output_dir or (settings.resolved_reports_dir() / "m1_baselines")
    seed = args.seed if args.seed is not None else settings.random_seed

    result = run_baseline(
        data_root=data_root,
        output_dir=output_dir,
        experiments_md=PROJECT_ROOT / "experiments.md",
        selector_name=args.selector,
        constant_value=args.constant_value,
        split=Split(args.split),
        language=Language(args.language),
        seed=seed,
    )

    logger.info("Prediction CSV: %s", result.prediction_csv)
    if not result.format_check_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
