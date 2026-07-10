"""Seeded random sampling and a contact-sheet visualization.

Two things the project's data-analysis notes flagged as real traps here:

1. `train`/`dev` JSONL rows are sorted alphabetically by country -- taking
   "the first N" would silently sample Algeria only. Every sample drawn
   here uses `random.Random(seed)`, never a slice.
2. Images include JPEG, PNG, and a few animated GIFs. The grid renderer
   normalizes all three to one static thumbnail size without assuming a
   single codec, and never lets one unreadable file abort the whole sheet.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ayn_vqa.data.schema import Task1aRecord

_THUMB_SIZE = (200, 200)
_PADDING = 8
_CAPTION_HEIGHT = 34
_BACKGROUND = (24, 24, 24)
_TEXT_COLOR = (230, 230, 230)
_MISSING_COLOR = (120, 30, 30)


def sample_records(records: list[Task1aRecord], n: int, seed: int) -> list[Task1aRecord]:
    """`random.Random(seed).sample`, never `records[:n]` -- see module
    docstring for why a slice would be a silently biased sample here."""
    rng = random.Random(seed)
    n = min(n, len(records))
    return rng.sample(records, n)


def _caption_for(record: Task1aRecord) -> str:
    parts = [record.id[:10]]
    if record.country:
        parts.append(record.country)
    if record.label is not None:
        parts.append(f"label={record.label}")
    return " | ".join(parts)


def _load_thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail(_THUMB_SIZE, Image.Resampling.LANCZOS)
        # Paste onto a fixed-size canvas so every grid cell lines up
        # regardless of the source image's aspect ratio.
        canvas = Image.new("RGB", _THUMB_SIZE, _BACKGROUND)
        offset = ((_THUMB_SIZE[0] - rgb.width) // 2, (_THUMB_SIZE[1] - rgb.height) // 2)
        canvas.paste(rgb, offset)
        return canvas


def _placeholder_tile(message: str) -> Image.Image:
    tile = Image.new("RGB", _THUMB_SIZE, _MISSING_COLOR)
    draw = ImageDraw.Draw(tile)
    draw.text((10, _THUMB_SIZE[1] // 2 - 20), message, fill=_TEXT_COLOR)
    return tile


def render_sample_grid(
    records: list[Task1aRecord], data_root: Path, out_path: Path, n_cols: int = 6
) -> Path:
    """Build a contact sheet: one tile per record, thumbnail + a one-line
    caption (id prefix, country, label). A missing or corrupt image renders
    as a labeled placeholder tile instead of raising -- an audit tool
    should make broken data visible, not crash on it.
    """
    n_rows = (len(records) + n_cols - 1) // n_cols
    cell_w = _THUMB_SIZE[0] + _PADDING
    cell_h = _THUMB_SIZE[1] + _CAPTION_HEIGHT + _PADDING
    sheet = Image.new("RGB", (n_cols * cell_w + _PADDING, n_rows * cell_h + _PADDING), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, record in enumerate(records):
        row, col = divmod(idx, n_cols)
        x = _PADDING + col * cell_w
        y = _PADDING + row * cell_h
        try:
            tile = _load_thumbnail(record.image_path(data_root))
        except Exception as exc:
            tile = _placeholder_tile(f"unreadable\n{exc}"[:60])
        sheet.paste(tile, (x, y))
        draw.text((x, y + _THUMB_SIZE[1] + 2), _caption_for(record), fill=_TEXT_COLOR, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
