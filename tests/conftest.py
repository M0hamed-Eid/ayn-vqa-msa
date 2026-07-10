"""Shared pytest fixtures.

`mini_dataset` builds a tiny synthetic dataset that mirrors the real Task1a
schema exactly (same directory layout, same JSONL field sets per split) so
every test runs in milliseconds without touching the multi-gigabyte real
dataset. It's generated in code rather than checked in as binary fixture
files, so it can never silently drift out of sync with `Task1aRecord`.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import soundfile as sf
from PIL import Image


def _make_noise_image(path: Path, seed: int, size: tuple[int, int] = (64, 48)) -> None:
    """A per-seed random-noise image: unlike a flat solid color, this has
    real pixel-to-pixel variation, which `dhash` (Ayn.audit.hashing) needs
    to produce a non-degenerate, content-distinguishing fingerprint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pixels = [
        (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for _ in range(size[0] * size[1])
    ]
    img = Image.new("RGB", size)
    img.putdata(pixels)
    img.save(path)


def _make_wav(path: Path, duration_sec: float, sample_rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration_sec * sample_rate)
    mono = [0.1 * math.sin(2 * math.pi * 440 * i / sample_rate) for i in range(n_frames)]
    data = mono if channels == 1 else [(s, s) for s in mono]
    sf.write(str(path), data, sample_rate)


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """Builds, under `tmp_path`, a dataset root with:

    - `images/`: 5 distinct random-noise JPEGs (`id00`-`id04`) plus a
      byte-identical copy of `id00` saved as `dup-exact` (exact-duplicate
      test case).
    - `audio/msa/`: one WAV per record, each a different duration (and
      `id01` rendered in stereo, to exercise channel-count probing).
    - `task1a/train_msa.jsonl`: 7 valid rows (including one, `missing-audio`,
      whose audio path is never created -- a missing-media test case) plus
      one malformed JSON line (a parse-error test case).
    - `task1a/dev_msa.jsonl`, `devtest_msa.jsonl`: minimal valid splits;
      `devtest` rows omit `label`/`country`/`category`/`subcategory`,
      matching the real dataset's schema.

    Returns the dataset root path (what production code calls `data_root`).
    """
    root = tmp_path
    ids = [f"id{i:02d}" for i in range(5)]

    for i, record_id in enumerate(ids):
        _make_noise_image(root / "images" / f"{record_id}.jpg", seed=i)
        _make_wav(
            root / "audio" / "msa" / f"{record_id}.wav",
            duration_sec=1.0 + i,
            channels=2 if i == 1 else 1,
        )

    dup_id = "dup-exact"
    (root / "images" / f"{dup_id}.jpg").write_bytes((root / "images" / "id00.jpg").read_bytes())
    _make_wav(root / "audio" / "msa" / f"{dup_id}.wav", duration_sec=2.5)

    countries = ["Egypt", "Jordan", "Morocco", "Sudan", "Tunisia", "Yemen"]
    all_ids = [*ids, dup_id]
    train_rows = [
        {
            "id": record_id,
            "image": f"images/{record_id}.jpg",
            "audio": f"audio/msa/{record_id}.wav",
            "country": country,
            "category": "Food & Cooking",
            "subcategory": "Dishes",
            "label": i % 3,
        }
        for i, (record_id, country) in enumerate(zip(all_ids, countries, strict=True))
    ]
    # Its own distinct image (not one of the ids/ folder above) so this row
    # tests exactly one thing -- a missing audio file -- without also
    # becoming a spurious member of the exact-duplicate-image test case.
    _make_noise_image(root / "images" / "missing-audio.jpg", seed=99)
    train_rows.append(
        {
            "id": "missing-audio",
            "image": "images/missing-audio.jpg",
            "audio": "audio/msa/does-not-exist.wav",
            "country": "Egypt",
            "category": "Food & Cooking",
            "subcategory": "Dishes",
            "label": 0,
        }
    )

    task1a_dir = root / "task1a"
    task1a_dir.mkdir(parents=True, exist_ok=True)

    with (task1a_dir / "train_msa.jsonl").open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
        f.write("{this is not valid json\n")

    with (task1a_dir / "dev_msa.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(train_rows[0]) + "\n")

    devtest_row = {
        "id": ids[0],
        "image": f"images/{ids[0]}.jpg",
        "audio": f"audio/msa/{ids[0]}.wav",
    }
    with (task1a_dir / "devtest_msa.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(devtest_row) + "\n")

    return root
