"""Image statistics: resolution, format, color mode, animation.

`Image.open` is lazy: it reads the header and defers full pixel decoding
until you call `.load()`, `.resize()`, etc. Probing thousands of images for
just width/height/format/mode is therefore fast and doesn't require
decoding every JPEG.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ImageStat:
    """Probe result for one image file. `error` is set (all other fields
    `None`/defaults) when the file is missing or Pillow can't decode it.
    """

    record_id: str
    path: Path
    width: int | None
    height: int | None
    format: str | None
    mode: str | None
    file_size_bytes: int | None
    is_animated: bool
    n_frames: int
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def megapixels(self) -> float | None:
        if self.width is None or self.height is None:
            return None
        return self.width * self.height / 1_000_000


def probe_image(record_id: str, path: Path) -> ImageStat:
    if not path.is_file():
        return ImageStat(record_id, path, None, None, None, None, None, False, 0, "file not found")

    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format
            mode = img.mode
            # Only GIF (and some TIFF/WEBP) ever set these; plain JPEG/PNG
            # fall back to the "one static frame" defaults via getattr.
            is_animated = bool(getattr(img, "is_animated", False))
            n_frames = int(getattr(img, "n_frames", 1))
        file_size_bytes = path.stat().st_size
    except Exception as exc:
        # Truncated downloads, zero-byte files, and decoder-bomb guards all
        # land here; the audit records the failure instead of crashing.
        return ImageStat(record_id, path, None, None, None, None, None, False, 0, str(exc))

    return ImageStat(
        record_id=record_id,
        path=path,
        width=width,
        height=height,
        format=fmt,
        mode=mode,
        file_size_bytes=file_size_bytes,
        is_animated=is_animated,
        n_frames=n_frames,
        error=None,
    )


def probe_many(items: Iterable[tuple[str, Path]]) -> list[ImageStat]:
    return [probe_image(record_id, path) for record_id, path in items]
