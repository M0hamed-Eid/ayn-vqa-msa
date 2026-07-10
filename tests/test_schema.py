from pathlib import Path

import pytest
from pydantic import ValidationError

from ayn_vqa.data.schema import Task1aRecord


def test_full_record_parses() -> None:
    record = Task1aRecord.model_validate(
        {
            "id": "abc123",
            "image": "images/abc123.jpg",
            "audio": "audio/msa/abc123.wav",
            "country": "Egypt",
            "category": "Food & Cooking",
            "subcategory": "Dishes",
            "label": 2,
        }
    )
    assert record.label == 2
    assert record.country == "Egypt"


def test_devtest_style_record_omits_optional_fields() -> None:
    record = Task1aRecord.model_validate(
        {"id": "abc123", "image": "images/abc123.jpg", "audio": "audio/msa/abc123.wav"}
    )
    assert record.label is None
    assert record.country is None
    assert record.category is None
    assert record.subcategory is None


@pytest.mark.parametrize("bad_label", [3, -1, 100])
def test_label_out_of_range_is_rejected(bad_label: int) -> None:
    with pytest.raises(ValidationError):
        Task1aRecord.model_validate(
            {"id": "x", "image": "images/x.jpg", "audio": "audio/msa/x.wav", "label": bad_label}
        )


def test_unexpected_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Task1aRecord.model_validate(
            {
                "id": "x",
                "image": "images/x.jpg",
                "audio": "audio/msa/x.wav",
                "surprise_field": "should not exist on any known split",
            }
        )


def test_media_paths_resolve_relative_to_data_root() -> None:
    record = Task1aRecord.model_validate(
        {"id": "x", "image": "images/x.jpg", "audio": "audio/msa/x.wav"}
    )
    data_root = Path("/data/AynVQA-ArabicNLP26")
    assert record.image_path(data_root) == data_root / "images" / "x.jpg"
    assert record.audio_path(data_root) == data_root / "audio" / "msa" / "x.wav"
