from pathlib import Path

from ayn_vqa.audit.hashing import (
    dhash,
    find_near_duplicates,
    group_exact_duplicates,
    hamming_distance,
    sha256_file,
)


def test_sha256_is_deterministic_and_content_sensitive(mini_dataset: Path) -> None:
    id00 = mini_dataset / "images" / "id00.jpg"
    dup = mini_dataset / "images" / "dup-exact.jpg"  # byte-identical copy, see conftest
    id01 = mini_dataset / "images" / "id01.jpg"

    assert sha256_file(id00) == sha256_file(id00)  # deterministic
    assert sha256_file(id00) == sha256_file(dup)  # byte-identical files hash equal
    assert sha256_file(id00) != sha256_file(id01)  # different content hashes differ


def test_group_exact_duplicates_finds_only_the_true_group() -> None:
    id_to_hash = {"a": "hash1", "b": "hash1", "c": "hash2", "d": "hash3"}
    groups = group_exact_duplicates(id_to_hash)

    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b"}


def test_dhash_is_a_stable_64_bit_fingerprint(mini_dataset: Path) -> None:
    path = mini_dataset / "images" / "id00.jpg"
    first = dhash(path)
    second = dhash(path)

    assert first == second
    assert 0 <= first < 2**64


def test_hamming_distance_counts_differing_bits() -> None:
    assert hamming_distance(0b1010, 0b1010) == 0
    assert hamming_distance(0b1010, 0b1000) == 1
    assert hamming_distance(0b0000, 0b1111) == 4


def test_find_near_duplicates_respects_threshold() -> None:
    a = 0b0000000000
    b = 0b0000000011  # popcount(a ^ b) == 2
    c = 0b1111111111  # popcount(a ^ c) == 10, popcount(b ^ c) == 8 -- far from both

    pairs = find_near_duplicates({"a": a, "b": b, "c": c}, max_distance=4)

    found = {frozenset((p.id_a, p.id_b)) for p in pairs}
    assert frozenset({"a", "b"}) in found
    assert frozenset({"a", "c"}) not in found
    assert frozenset({"b", "c"}) not in found
