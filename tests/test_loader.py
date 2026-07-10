from pathlib import Path

import pytest

from ayn_vqa.data.loader import load_all_splits, load_split
from ayn_vqa.data.schema import Language, Split


def test_load_split_separates_valid_records_from_errors(mini_dataset: Path) -> None:
    split_data = load_split(mini_dataset, Split.TRAIN, Language.MSA)

    assert len(split_data) == 7  # 5 ids + dup-exact + missing-audio
    assert len(split_data.errors) == 1  # the one malformed JSON line
    assert split_data.errors[0].line_number == 8


def test_devtest_records_have_no_label(mini_dataset: Path) -> None:
    split_data = load_split(mini_dataset, Split.DEVTEST, Language.MSA)
    assert len(split_data) == 1
    assert split_data.records[0].label is None


def test_missing_split_file_raises(mini_dataset: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split(mini_dataset, Split.TRAIN, Language.EN)


def test_load_all_splits_returns_one_entry_per_split(mini_dataset: Path) -> None:
    result = load_all_splits(mini_dataset, language=Language.MSA)
    assert set(result.keys()) == {Split.TRAIN, Split.DEV, Split.DEVTEST}
    assert len(result[Split.DEV]) == 1
