"""Aggregate everything the audit computed into CSV, JSON, and Markdown
reports, plus a handful of PNG figures.

Design choice: every intermediate result (records, media checks, audio/image
stats) is a plain dataclass produced by another module; this module's only
job is to fold them into `pandas.DataFrame`s (for the tabular exports) and a
plain nested `dict` (for the JSON summary + Markdown tables). It holds no
probing logic of its own -- that separation is what makes each half unit
testable independently (fake a DataFrame vs. fake a WAV file).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this runs from a CLI/notebook kernel, never a GUI session

import matplotlib.pyplot as plt
import pandas as pd

from ayn_vqa.audit.audio_stats import AudioStat
from ayn_vqa.audit.hashing import NearDuplicatePair
from ayn_vqa.audit.image_stats import ImageStat
from ayn_vqa.data.loader import SplitData
from ayn_vqa.data.validation import MediaCheck

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitAuditData:
    """Everything computed for one (split, language) pair, bundled up for
    reporting."""

    split_data: SplitData
    media_checks: list[MediaCheck]
    audio_stats: list[AudioStat]
    image_stats: list[ImageStat]


@dataclass(frozen=True)
class ReportPaths:
    manifest_csv: Path
    duplicates_csv: Path
    summary_json: Path
    report_md: Path
    figures: list[Path]
    sample_grid: Path | None


def _manifest_frame(bundles: list[SplitAuditData]) -> pd.DataFrame:
    """One row per record: split/language, media existence, image + audio
    stats side by side. This becomes `file_manifest.csv` -- the one table
    you'd open to answer "what's wrong with item X".

    Every record has exactly one `MediaCheck` and one stat object per media
    field, even when the file is missing (the probing functions return an
    error-flagged result rather than being skipped) -- so no `None`-handling
    is needed here.
    """
    rows: list[dict[str, Any]] = []

    for bundle in bundles:
        audio_by_id = {s.record_id: s for s in bundle.audio_stats}
        image_by_id = {s.record_id: s for s in bundle.image_stats}
        media_by_id_field = {(c.record_id, c.field): c for c in bundle.media_checks}

        for record in bundle.split_data.records:
            image_check = media_by_id_field[(record.id, "image")]
            audio_check = media_by_id_field[(record.id, "audio")]
            image_stat = image_by_id[record.id]
            audio_stat = audio_by_id[record.id]
            rows.append(
                {
                    "id": record.id,
                    "split": bundle.split_data.split.value,
                    "language": bundle.split_data.language.value,
                    "country": record.country,
                    "category": record.category,
                    "subcategory": record.subcategory,
                    "label": record.label,
                    "image_relpath": record.image,
                    "image_exists": image_check.exists,
                    "image_width": image_stat.width,
                    "image_height": image_stat.height,
                    "image_format": image_stat.format,
                    "image_mode": image_stat.mode,
                    "image_is_animated": image_stat.is_animated,
                    "image_size_bytes": image_stat.file_size_bytes,
                    "image_error": image_stat.error,
                    "audio_relpath": record.audio,
                    "audio_exists": audio_check.exists,
                    "audio_duration_sec": audio_stat.duration_sec,
                    "audio_sample_rate": audio_stat.sample_rate,
                    "audio_channels": audio_stat.channels,
                    "audio_subtype": audio_stat.subtype,
                    "audio_size_bytes": audio_check.size_bytes,
                    "audio_error": audio_stat.error,
                }
            )

    manifest = pd.DataFrame(rows)
    # Pandas upcasts an int column containing `None` (e.g. `label` on
    # unlabeled devtest rows) to float64, so 0/1/2 would otherwise print as
    # "0.0"/"1.0"/"2.0" in the report. The nullable "Int64" extension dtype
    # keeps missing values as `pd.NA` while formatting real values as plain
    # integers.
    manifest["label"] = manifest["label"].astype("Int64")
    return manifest


def _duplicates_frame(
    exact_groups: list[list[str]], near_pairs: list[NearDuplicatePair]
) -> pd.DataFrame:
    """One row per duplicate *relationship*. For an exact-duplicate group of
    size k, that's k-1 rows (anchor vs. each other member) -- enough to
    reconstruct the full equivalence class without an O(k^2) blow-up for
    large groups.
    """
    rows: list[dict[str, Any]] = []
    for group in exact_groups:
        sorted_ids = sorted(group)
        anchor, others = sorted_ids[0], sorted_ids[1:]
        for other in others:
            rows.append({"kind": "exact", "id_a": anchor, "id_b": other, "hamming_distance": 0})
    for pair in near_pairs:
        rows.append(
            {
                "kind": "near",
                "id_a": pair.id_a,
                "id_b": pair.id_b,
                "hamming_distance": pair.distance,
            }
        )
    return pd.DataFrame(rows, columns=["kind", "id_a", "id_b", "hamming_distance"])


def _counts(series: pd.Series) -> dict[str, int]:
    """`value_counts` as a plain `{label: count}` dict with string keys --
    sidesteps numpy scalar keys/values that `json.dumps` can't serialize."""
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def build_summary(
    manifest: pd.DataFrame, duplicates: pd.DataFrame, parse_errors_by_split: dict[str, int]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_records": int(len(manifest)),
        "parse_errors_by_split": parse_errors_by_split,
        "media": {
            "images_missing": int((~manifest["image_exists"]).sum()),
            "audio_missing": int((~manifest["audio_exists"]).sum()),
            "images_unreadable": int(manifest["image_error"].notna().sum()),
            "audio_unreadable": int(manifest["audio_error"].notna().sum()),
        },
        "by_split_language": (
            manifest.groupby(["split", "language"])
            .size()
            .rename("n_records")
            .reset_index()
            .to_dict(orient="records")
        ),
        "image_formats": _counts(manifest["image_format"]),
        "animated_gif_count": int(manifest["image_is_animated"].fillna(False).sum()),
        "image_width_stats": manifest["image_width"].dropna().describe().to_dict(),
        "image_height_stats": manifest["image_height"].dropna().describe().to_dict(),
        "audio_sample_rates": _counts(manifest["audio_sample_rate"]),
        "audio_channels": _counts(manifest["audio_channels"]),
        "audio_duration_sec_stats": manifest["audio_duration_sec"].dropna().describe().to_dict(),
        "label_distribution": _counts(manifest["label"]),
        "country_distribution": _counts(manifest["country"]),
        "category_distribution": _counts(manifest["category"]),
        "duplicates": {
            "exact_duplicate_pairs": int((duplicates["kind"] == "exact").sum())
            if not duplicates.empty
            else 0,
            "near_duplicate_pairs": int((duplicates["kind"] == "near").sum())
            if not duplicates.empty
            else 0,
        },
    }
    return summary


def _to_native(value: Any) -> Any:
    """Recursively convert a nested dict/list produced by pandas
    aggregations into plain, `json.dumps`-safe Python: numpy scalars via
    `.item()`, NaN -> `None`, every dict key -> `str`.
    """
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def render_figures(manifest: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Save the handful of PNG charts the Markdown report and notebook both
    embed. Each figure is skipped (not an error) if its column is entirely
    empty, e.g. no audio stats at all because every file was missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    durations = manifest["audio_duration_sec"].dropna()
    if not durations.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(durations, bins=40)
        ax.set_xlabel("duration (s)")
        ax.set_ylabel("count")
        ax.set_title("Audio duration distribution")
        path = out_dir / "audio_duration_hist.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    widths = manifest["image_width"].dropna()
    heights = manifest["image_height"].dropna()
    if not widths.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(widths, heights, s=6, alpha=0.3)
        ax.set_xlabel("width (px)")
        ax.set_ylabel("height (px)")
        ax.set_title("Image resolution scatter")
        path = out_dir / "image_resolution_scatter.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    formats = manifest["image_format"].value_counts(dropna=False)
    if not formats.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        formats.plot(kind="bar", ax=ax)
        ax.set_ylabel("count")
        ax.set_title("Image format counts")
        path = out_dir / "image_format_counts.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    countries = manifest["country"].dropna().value_counts()
    if not countries.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        countries.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("count")
        ax.set_title("Country distribution (records with metadata)")
        path = out_dir / "country_counts.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def render_markdown(
    summary: dict[str, Any],
    duplicates: pd.DataFrame,
    figures: list[Path],
    sample_grid_path: Path | None,
    out_path: Path,
    out_dir: Path,
) -> Path:
    figure_names = {p.name for p in figures}
    lines: list[str] = ["# M0 Data Audit Report", ""]
    lines.append(f"Total records audited: **{summary['total_records']}**")
    lines.append("")

    lines.append("## Records per split/language")
    lines.append(pd.DataFrame(summary["by_split_language"]).to_markdown(index=False))
    lines.append("")

    media = summary["media"]
    lines += [
        "## Media integrity",
        f"- Missing images: **{media['images_missing']}**",
        f"- Missing audio: **{media['audio_missing']}**",
        f"- Present-but-unreadable images: **{media['images_unreadable']}**",
        f"- Present-but-unreadable audio: **{media['audio_unreadable']}**",
        "",
    ]
    if summary["parse_errors_by_split"]:
        lines.append("### JSONL parse errors by split")
        lines += [f"- `{key}`: {count}" for key, count in summary["parse_errors_by_split"].items()]
        lines.append("")

    lines += [
        "## Image statistics",
        f"- Formats: `{summary['image_formats']}`",
        f"- Animated GIFs: {summary['animated_gif_count']}",
        "",
    ]
    if "image_resolution_scatter.png" in figure_names:
        lines += ["![Image resolution scatter](figures/image_resolution_scatter.png)", ""]
    if "image_format_counts.png" in figure_names:
        lines += ["![Image format counts](figures/image_format_counts.png)", ""]

    lines += [
        "## Audio statistics",
        f"- Sample rates: `{summary['audio_sample_rates']}`",
        f"- Channels: `{summary['audio_channels']}`",
    ]
    duration_stats = summary["audio_duration_sec_stats"]
    if duration_stats:
        lines.append(
            f"- Duration (s): mean={duration_stats.get('mean', float('nan')):.2f}, "
            f"median={duration_stats.get('50%', float('nan')):.2f}, "
            f"min={duration_stats.get('min', float('nan')):.2f}, "
            f"max={duration_stats.get('max', float('nan')):.2f}"
        )
    lines.append("")
    if "audio_duration_hist.png" in figure_names:
        lines += ["![Audio duration histogram](figures/audio_duration_hist.png)", ""]

    lines += [
        "## Label / country / category distribution",
        f"- Labels: `{summary['label_distribution']}`",
        "",
    ]
    if "country_counts.png" in figure_names:
        lines += ["![Country distribution](figures/country_counts.png)", ""]

    n_exact = summary["duplicates"]["exact_duplicate_pairs"]
    n_near = summary["duplicates"]["near_duplicate_pairs"]
    lines += [
        "## Duplicate images",
        f"- Exact byte-identical duplicate pairs: **{n_exact}**",
        f"- Near-duplicate pairs (perceptual hash): **{n_near}**",
        "- Full list: `duplicate_images.csv`",
        "",
    ]

    if sample_grid_path is not None:
        rel = sample_grid_path.relative_to(out_dir)
        lines += ["## Random sample", f"![Random sample grid]({rel.as_posix()})", ""]

    lines += [
        "## Raw data",
        "- `file_manifest.csv` -- one row per record with every stat computed above",
        "- `duplicate_images.csv` -- every exact/near duplicate pair found",
        "- `audit_summary.json` -- this report's numbers as machine-readable JSON",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_reports(
    bundles: list[SplitAuditData],
    exact_dup_groups: list[list[str]],
    near_dup_pairs: list[NearDuplicatePair],
    out_dir: Path,
    sample_grid_path: Path | None = None,
) -> ReportPaths:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest_frame(bundles)
    duplicates = _duplicates_frame(exact_dup_groups, near_dup_pairs)

    manifest_csv = out_dir / "file_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)

    duplicates_csv = out_dir / "duplicate_images.csv"
    duplicates.to_csv(duplicates_csv, index=False)

    parse_errors_by_split = {
        f"{b.split_data.split.value}_{b.split_data.language.value}": len(b.split_data.errors)
        for b in bundles
    }
    summary = build_summary(manifest, duplicates, parse_errors_by_split)

    summary_json = out_dir / "audit_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(_to_native(summary), f, indent=2, ensure_ascii=False)

    figures = render_figures(manifest, out_dir / "figures")

    report_md = out_dir / "audit_report.md"
    render_markdown(summary, duplicates, figures, sample_grid_path, report_md, out_dir)

    logger.info("Wrote audit reports to %s", out_dir)
    return ReportPaths(
        manifest_csv=manifest_csv,
        duplicates_csv=duplicates_csv,
        summary_json=summary_json,
        report_md=report_md,
        figures=figures,
        sample_grid=sample_grid_path,
    )
