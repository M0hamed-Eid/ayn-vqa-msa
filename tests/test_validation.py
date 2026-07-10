from pathlib import Path

from ayn_vqa.data.loader import load_split
from ayn_vqa.data.schema import Language, Split
from ayn_vqa.data.validation import missing_media, validate_split_media


def test_validate_split_media_flags_only_the_missing_audio_file(mini_dataset: Path) -> None:
    split_data = load_split(mini_dataset, Split.TRAIN, Language.MSA)
    checks = validate_split_media(split_data, mini_dataset)

    assert len(checks) == len(split_data) * 2  # image + audio per record

    missing = missing_media(checks)
    assert len(missing) == 1
    assert missing[0].record_id == "missing-audio"
    assert missing[0].field == "audio"


def test_existing_files_report_size_bytes(mini_dataset: Path) -> None:
    split_data = load_split(mini_dataset, Split.TRAIN, Language.MSA)
    checks = validate_split_media(split_data, mini_dataset)

    image_checks = [c for c in checks if c.field == "image" and c.exists]
    assert image_checks
    assert all(c.size_bytes is not None and c.size_bytes > 0 for c in image_checks)
