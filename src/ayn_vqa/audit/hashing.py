"""Exact and near-duplicate detection over image bytes.

Two independent notions of "duplicate", because they catch different bugs:

- **Exact** (SHA-256 of the raw file bytes): the same file was copied under
  two ids -- e.g. accidentally reused across train/dev/devtest. Cheap,
  O(n), zero false positives.
- **Near** (a perceptual difference-hash, "dHash"): the same *picture*
  re-encoded, resized, or re-compressed under two ids -- something exact
  hashing cannot catch, because re-saving an image changes every byte even
  though a human sees the same photo. dHash reduces the pixel content to a
  64-bit fingerprint that's stable under those transformations; two images
  are "near duplicates" if their fingerprints differ in only a few bits
  (Hamming distance).

At this dataset's scale (~4,000 images per language track), an O(n^2)
all-pairs Hamming comparison is a few million cheap integer ops -- a couple
of seconds in CPython using `int.bit_count()`. An indexed structure (BK-tree,
LSH) would be the right call at 10-100x this size; building one here would
be premature engineering for data this small.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_DHASH_SIZE = 8  # 8x8 grid of comparisons -> 64-bit fingerprint


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streamed so multi-megabyte images don't need to fit in memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, hash_size: int = _DHASH_SIZE) -> int:
    """Difference hash (Neal Krawetz's dHash): shrink to a tiny grayscale
    grid, compare each pixel to its right neighbor, pack the booleans into
    an integer. Robust to resizing, re-compression, and minor color shifts
    -- exactly the transformations that would defeat a byte-exact hash on a
    re-saved duplicate -- while needing only Pillow, no extra dependency.
    """
    with Image.open(path) as img:
        # (hash_size + 1) columns so every one of the `hash_size` columns
        # has a right-hand neighbor to compare against.
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        # `tobytes()` on a single-band ("L") image is one byte per pixel,
        # row-major -- the same ordering as `getdata()` without the
        # deprecation (removal slated for Pillow 14).
        pixels = small.tobytes()

    bits = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            bits <<= 1
            if pixels[row_start + col] > pixels[row_start + col + 1]:
                bits |= 1
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def group_exact_duplicates(id_to_hash: dict[str, str]) -> list[list[str]]:
    """Group ids that share an identical SHA-256, i.e. byte-identical files."""
    groups: dict[str, list[str]] = {}
    for record_id, digest in id_to_hash.items():
        groups.setdefault(digest, []).append(record_id)
    return [ids for ids in groups.values() if len(ids) > 1]


@dataclass(frozen=True)
class NearDuplicatePair:
    id_a: str
    id_b: str
    distance: int


def find_near_duplicates(
    id_to_phash: dict[str, int], max_distance: int = 4
) -> list[NearDuplicatePair]:
    """All pairs within `max_distance` bits of each other (out of 64).

    See the module docstring for why plain O(n^2) is the right trade-off at
    this dataset's size rather than an indexed nearest-neighbor structure.
    """
    ids = list(id_to_phash.keys())
    pairs: list[NearDuplicatePair] = []
    for i in range(len(ids)):
        hash_i = id_to_phash[ids[i]]
        for j in range(i + 1, len(ids)):
            distance = hamming_distance(hash_i, id_to_phash[ids[j]])
            if distance <= max_distance:
                pairs.append(NearDuplicatePair(ids[i], ids[j], distance))
    return pairs
