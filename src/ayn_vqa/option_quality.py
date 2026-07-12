"""Option-quality validation: catches parsed MCQ options that are
structurally well-formed (the JSON schema always succeeds) but
semantically degenerate.

`OllamaTranscriptParser`'s schema forces exactly three non-empty option
strings every time. When the underlying transcript doesn't actually
contain three cleanly recoverable options -- ASR dropped one, or the
parser mis-split a run-on sentence -- it fabricates content instead of
signaling failure: an empty string, the literal preamble phrase
"الخيارات هي" ("the options are") captured as if it were an option, a
duplicate of another option, or placeholder text like "الخيار الأول"
("the first option").

`error_analysis.py`'s `asr_error`/`parse_error` columns don't catch this --
they only fire on a hard exception or an empty transcript. Hand-reading all
142 wrong M3 dev predictions found that 69 of them (48.6%) are exactly this
kind of silent pipeline corruption, not genuine VLM visual-reasoning
misses; the checks below are those same hand-verified heuristics, not new
guesses. No Protocol here -- there is one rule-based implementation, not
several swappable backends, so a swappable interface would be premature.
"""

from __future__ import annotations

from dataclasses import dataclass

_MIN_OPTION_LEN = 3

# Leftover-preamble substrings, as actually observed in ASR transcripts:
# the parser sometimes captures the "the options are" lead-in itself as if
# it were the third option, in whichever spelling variant Whisper produced.
_BOILERPLATE_MARKERS: tuple[str, ...] = (
    "الخيارات هي",
    "الخيارات وياه",
    "القيارات",
    "الاختيارات",
    "اللخيارات",
    "الخيار الأول",
    "الخيار الثاني",
    "الخيار الثالث",
)

# Fabricated "no answer" placeholders the parser produces when it has no
# real third option to extract, as actually observed in parsed output.
_NO_ANSWER_MARKERS: tuple[str, ...] = (
    "لا يوجد",
    "غير مناسب",
    "غير متوفرة",
    "غير صحيح",
)


@dataclass(frozen=True)
class OptionQualityCheck:
    """`reasons` is empty iff `is_degenerate` is `False`. Multiple reasons
    can co-occur (e.g. one option empty and two others duplicated)."""

    is_degenerate: bool
    reasons: tuple[str, ...]


def _single_option_reasons(option: str) -> set[str]:
    text = option.strip()
    reasons: set[str] = set()
    if len(text) < _MIN_OPTION_LEN:
        reasons.add("empty_option")
        return reasons  # too short to meaningfully contain the markers below
    if any(marker in text for marker in _BOILERPLATE_MARKERS):
        reasons.add("boilerplate_leak")
    if any(marker in text for marker in _NO_ANSWER_MARKERS):
        reasons.add("fabricated_placeholder")
    return reasons


def _cross_option_reasons(options: tuple[str, str, str]) -> set[str]:
    stripped = [o.strip() for o in options]
    substantial = [s for s in stripped if len(s) >= _MIN_OPTION_LEN]

    reasons: set[str] = set()
    if len(substantial) != len(set(substantial)):
        reasons.add("duplicate_option")

    for i, a in enumerate(stripped):
        if len(a) < _MIN_OPTION_LEN:
            continue
        for j, b in enumerate(stripped):
            if i != j and a != b and len(b) >= _MIN_OPTION_LEN and a in b:
                reasons.add("overlapping_option")
    return reasons


def check_option_quality(options: tuple[str, str, str]) -> OptionQualityCheck:
    reasons: set[str] = set()
    for option in options:
        reasons |= _single_option_reasons(option)
    reasons |= _cross_option_reasons(options)
    return OptionQualityCheck(is_degenerate=bool(reasons), reasons=tuple(sorted(reasons)))


def check_repetition_loop(
    transcript_text: str, min_repeats: int = 5, max_token_len: int = 2
) -> bool:
    """True if a short token (e.g. a single Arabic letter) repeats
    `min_repeats`+ times in a row -- the degenerate-repetition pattern
    Whisper produces on unclear/noisy audio (observed: 32 consecutive
    repetitions of the single letter "آ" on one M3 dev item). Operates on
    the raw transcript, not parsed options, since this corruption
    originates in ASR, not in parsing.
    """
    tokens = transcript_text.split()
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1] and len(tokens[i]) <= max_token_len:
            run += 1
            if run >= min_repeats:
                return True
        else:
            run = 1
    return False
