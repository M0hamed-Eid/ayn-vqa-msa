from __future__ import annotations

from ayn_vqa.option_quality import check_option_quality, check_repetition_loop


def test_three_clean_distinct_options_are_not_degenerate() -> None:
    result = check_option_quality(("الفن الإسلامي", "الفن التجريدي", "فن عصر النهضة"))

    assert not result.is_degenerate
    assert result.reasons == ()


def test_empty_option_is_flagged() -> None:
    result = check_option_quality(("خشب", "بلاستيك", ""))

    assert result.is_degenerate
    assert "empty_option" in result.reasons


def test_near_empty_option_is_flagged() -> None:
    result = check_option_quality(("خشب", "بلاستيك", "لا"))

    assert result.is_degenerate
    assert "empty_option" in result.reasons


def test_boilerplate_preamble_leak_is_flagged() -> None:
    # observed on a real M3 dev item: the "the options are" preamble
    # captured as if it were the third option.
    result = check_option_quality(
        (
            "المقياس التقليدي نحت خشبي منحوت على شكل ضفدع",
            "الأكواب الشاي السيراميك الأبيض",
            "الخيارات هي",
        )
    )

    assert result.is_degenerate
    assert "boilerplate_leak" in result.reasons


def test_ordinal_placeholder_options_are_flagged() -> None:
    # observed verbatim: parser had no real content to extract and
    # fabricated placeholder labels instead.
    result = check_option_quality(("الخيار الأول", "الخيار الثاني", "الخيار الثالث"))

    assert result.is_degenerate
    assert "boilerplate_leak" in result.reasons


def test_fabricated_no_answer_placeholder_is_flagged() -> None:
    result = check_option_quality(("حفل موسيقي", "تجمع سياسي", "لا يوجد"))

    assert result.is_degenerate
    assert "fabricated_placeholder" in result.reasons


def test_exact_duplicate_options_are_flagged() -> None:
    result = check_option_quality(("أمريكا الجنوبية", "الشرق", "الشرق"))

    assert result.is_degenerate
    assert "duplicate_option" in result.reasons


def test_overlapping_option_is_flagged() -> None:
    # one option's full text is a prefix/substring of another -- a parser
    # split-boundary defect, not two genuinely distinct choices.
    result = check_option_quality(
        (
            "الملابس والاكسورات الذهب والنجوهرات",
            "الملابس والاكسورات الذهب والنجوهرات الالكترونيات",
            "شيء آخر",
        )
    )

    assert result.is_degenerate
    assert "overlapping_option" in result.reasons


def test_multiple_reasons_can_co_occur() -> None:
    result = check_option_quality(("خيار حقيقي", "خيار حقيقي", ""))

    assert result.is_degenerate
    assert "duplicate_option" in result.reasons
    assert "empty_option" in result.reasons


def test_repetition_loop_absent_on_normal_transcript() -> None:
    transcript = "ما هو اللون الأكثر بروزا في الصورة الخيارات هي أحمر أزرق أخضر"

    assert not check_repetition_loop(transcript)


def test_repetition_loop_detected_on_repeated_short_token() -> None:
    # observed verbatim: 32 consecutive repetitions of the single letter
    # "آ" -- a degenerate ASR hallucination on unclear audio.
    prefix = "ما هي بطاقة الرسومات الموجودة في جهاز العبد هذا الخيارات هي "
    transcript = prefix + "آ " * 32 + "الثمانين"

    assert check_repetition_loop(transcript)


def test_repetition_loop_not_triggered_below_threshold() -> None:
    transcript = "سؤال الخيارات هي " + "آ " * 4 + "خيار آخر"

    assert not check_repetition_loop(transcript)


def test_repetition_loop_ignores_long_repeated_words() -> None:
    # a legitimately repeated real word (e.g. disfluency) shouldn't trip
    # this -- only short (<=2 char) token runs count.
    transcript = "خيار خيار خيار خيار خيار آخر"

    assert not check_repetition_loop(transcript)
