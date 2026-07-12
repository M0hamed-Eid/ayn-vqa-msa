"""Transcript parsing stage: raw ASR text -> {question, 3 options}.

Uses a local VLM (Qwen2.5-VL via Ollama, text-only for this stage -- no
image needed to split a transcript into its parts) with JSON-schema-
constrained output, so the model literally cannot return anything except
the shape this pipeline needs. No regex, no free-text parsing, no
"couldn't find a delimiter" failure mode. M2's ASR bench measured that the
literal phrase "الخيارات هي" ("the options are") appears in ~80% of
sampled transcripts (see docs/M2_ASR_BENCH.md) -- the prompt below
mentions it as a hint, not a hard rule the parser depends on, since the
other ~20% still need parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ayn_vqa.ollama_client import OllamaClient

_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "option_0": {"type": "string"},
        "option_1": {"type": "string"},
        "option_2": {"type": "string"},
    },
    "required": ["question", "option_0", "option_1", "option_2"],
}

_PROMPT = """The following is a speech-to-text transcript of a spoken Arabic \
multiple-choice question: one question, followed by exactly three answer \
options, read aloud in order (the first option spoken is index 0, the \
second is index 1, the third is index 2). The transcript may contain ASR \
errors -- do your best to recover the intended question and options. \
Often (not always) the options are introduced by the phrase "الخيارات هي" \
("the options are").

Transcript:
{transcript}

Split it into the question and the three options, in order."""

# Repair-retry variant, used only when a first parse of this same
# transcript produced degenerate options (see `ayn_vqa.option_quality`).
# No schema change -- still exactly {question, option_0, option_1,
# option_2} -- just a more pointed instruction about the specific failure
# mode hand-verified across the M3 dev error analysis: the parser
# sometimes captures the "options are" preamble itself as if it were an
# option, or duplicates/leaves one blank when it can't cleanly find three.
_STRICT_PROMPT = """The following is a speech-to-text transcript of a spoken \
Arabic multiple-choice question. A previous attempt at parsing this exact \
transcript produced a degenerate result -- an option that was empty, a \
duplicate of another option, or leftover phrasing like "الخيارات هي" \
("the options are") captured as if it were an option itself rather than \
genuine answer text. Look again, more carefully:

- The phrase "الخيارات هي" (or a similar variant) only ever introduces the \
options that follow -- it is never itself an option.
- The three options must be genuinely distinct from each other and from \
the question.
- If the transcript truly does not contain three clearly distinguishable \
options, give your best-effort guess at a plausible third option based on \
the question's subject matter, rather than repeating another option or \
leaving one blank.

Transcript:
{transcript}

Split it into the question and the three options, in order."""


@dataclass(frozen=True)
class ParsedTranscript:
    """`error` is set (`question`/`options` `None`) when the transcript was
    empty/unusable or the model call failed -- never raises.
    """

    record_id: str
    question: str | None
    options: tuple[str, str, str] | None
    backend_name: str
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


class TranscriptParser(Protocol):
    name: str

    def parse(self, record_id: str, transcript_text: str) -> ParsedTranscript: ...


class OllamaTranscriptParser:
    name = "ollama-parse"

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434",
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._ollama = OllamaClient(base_url=base_url, client=client)

    def parse(
        self, record_id: str, transcript_text: str, *, strict: bool = False
    ) -> ParsedTranscript:
        """`strict=True` selects the repair-retry prompt (see
        `_STRICT_PROMPT`) instead of the normal one -- an optional,
        defaulted keyword argument, so this still satisfies the plain
        `TranscriptParser` Protocol every other caller uses."""
        template = _STRICT_PROMPT if strict else _PROMPT
        try:
            content = self._ollama.chat(
                self._model,
                template.format(transcript=transcript_text),
                json_schema=_PARSE_SCHEMA,
            )
            payload = json.loads(content)
            options = (
                str(payload["option_0"]),
                str(payload["option_1"]),
                str(payload["option_2"]),
            )
            question = str(payload["question"])
        except Exception as exc:
            return ParsedTranscript(record_id, None, None, self.name, str(exc))
        return ParsedTranscript(record_id, question, options, self.name, None)
