"""End-to-end test of the M1 harness: load -> predict -> submit -> score,
against the synthetic `mini_dataset` fixture (not the real dataset) -- what
actually proves the modules compose, on top of each module's own tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ayn_vqa.data.schema import Language, Split
from ayn_vqa.run_baseline import run_baseline

# conftest.mini_dataset's train_msa.jsonl has 7 valid records with labels
# [0, 1, 2, 0, 1, 2, 0] (see tests/conftest.py) -- 3 of 7 are label 0.
_TRAIN_LABEL_0_FRACTION = 3 / 7


def test_constant_selector_matches_hand_computed_label0_fraction(
    mini_dataset: Path, tmp_path: Path
) -> None:
    result = run_baseline(
        data_root=mini_dataset,
        output_dir=tmp_path / "predictions",
        experiments_md=tmp_path / "experiments.md",
        selector_name="constant",
        constant_value=0,
        split=Split.TRAIN,
        language=Language.MSA,
        seed=42,
    )

    assert result.format_check_ok
    assert result.metrics is not None
    by_name = {m.name: m.value for m in result.metrics}
    assert by_name["accuracy"] == pytest.approx(_TRAIN_LABEL_0_FRACTION, abs=1e-6)

    with result.prediction_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    assert all(row["prediction"] == "0" for row in rows)


def test_random_selector_runs_end_to_end_and_logs_experiment(
    mini_dataset: Path, tmp_path: Path
) -> None:
    experiments_md = tmp_path / "experiments.md"

    result = run_baseline(
        data_root=mini_dataset,
        output_dir=tmp_path / "predictions",
        experiments_md=experiments_md,
        selector_name="random",
        constant_value=0,
        split=Split.TRAIN,
        language=Language.MSA,
        seed=7,
    )

    assert result.format_check_ok
    assert result.metrics is not None
    assert 0.0 <= dict((m.name, m.value) for m in result.metrics)["accuracy"] <= 1.0

    assert experiments_md.exists()
    log_text = experiments_md.read_text(encoding="utf-8")
    assert "random" in log_text
    assert "train" in log_text


def test_devtest_split_has_no_labels_and_is_not_scored(mini_dataset: Path, tmp_path: Path) -> None:
    result = run_baseline(
        data_root=mini_dataset,
        output_dir=tmp_path / "predictions",
        experiments_md=tmp_path / "experiments.md",
        selector_name="constant",
        constant_value=0,
        split=Split.DEVTEST,
        language=Language.MSA,
        seed=42,
    )

    assert result.format_check_ok
    assert result.metrics is None
