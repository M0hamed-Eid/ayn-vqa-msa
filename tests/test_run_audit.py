"""End-to-end test of the whole M0 pipeline: load -> validate -> probe ->
hash -> sample -> report, run against the synthetic `mini_dataset` fixture
instead of the real (multi-gigabyte) dataset. This is what actually proves
the modules compose correctly, on top of each module's own unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ayn_vqa.audit.run_audit import run_audit
from ayn_vqa.data.schema import Language, Split


def test_run_audit_end_to_end(mini_dataset: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "report_out"

    report_paths = run_audit(
        data_root=mini_dataset,
        output_dir=output_dir,
        language=Language.MSA,
        splits=(Split.TRAIN, Split.DEV, Split.DEVTEST),
        sample_n=4,
        near_dup_max_distance=4,
        seed=42,
    )

    assert report_paths.manifest_csv.exists()
    assert report_paths.duplicates_csv.exists()
    assert report_paths.summary_json.exists()
    assert report_paths.report_md.exists()
    assert report_paths.sample_grid is not None and report_paths.sample_grid.exists()

    manifest = pd.read_csv(report_paths.manifest_csv)
    # train (7) + dev (1) + devtest (1) records loaded by conftest.mini_dataset
    assert len(manifest) == 9
    assert manifest["audio_exists"].sum() == 8  # every record except "missing-audio"

    duplicates = pd.read_csv(report_paths.duplicates_csv)
    exact_rows = duplicates[duplicates["kind"] == "exact"]
    assert set(exact_rows[["id_a", "id_b"]].iloc[0]) == {"id00", "dup-exact"}

    summary = json.loads(report_paths.summary_json.read_text(encoding="utf-8"))
    assert summary["total_records"] == 9
    assert summary["media"]["audio_missing"] == 1
    assert summary["parse_errors_by_split"]["train_msa"] == 1
    assert summary["duplicates"]["exact_duplicate_pairs"] == 1

    report_text = report_paths.report_md.read_text(encoding="utf-8")
    assert "# M0 Data Audit Report" in report_text
