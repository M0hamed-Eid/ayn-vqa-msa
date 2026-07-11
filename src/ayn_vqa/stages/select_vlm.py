"""Joint-MCQ answer selection: image + question + 3 options -> predicted
index, via a local VLM (Qwen2.5-VL through Ollama).

"Joint" because all three options are shown to the model together in one
prompt, not scored independently one at a time. The project's plan doc
argues independent per-option entailment scoring discards the contrastive
signal MCQ distractors rely on -- "which option is MORE plausible than the
other two" is not the same question as "is this option plausible in
isolation", and only the joint framing can represent the former.

The predicted index is JSON-schema-constrained to exactly `{0, 1, 2}` --
the model cannot reply with free text, an Arabic digit, or an ordinal that
a downstream regex would then need to parse. The project's plan doc
specifically calls out the official baseline's ASCII-only `[012]` regex
fallback (silently defaulting to 0 on anything else, including a
perfectly correct Arabic-language answer) as likely inflating its
reported error rate. Constrained decoding doesn't parse around that bug
better -- it makes the bug's precondition (free-text output) structurally
impossible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from ayn_vqa.ollama_client import OllamaClient
from ayn_vqa.stages.select import Prediction

FALLBACK_INDEX = 0  # matches the organizers' own baseline fallback

_SELECT_SCHEMA = {
    "type": "object",
    "properties": {"answer_index": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["answer_index"],
}

_PROMPT = """You are given an image and a multiple-choice question about it, \
in Arabic. Using ONLY what is visible in the image, choose the option that \
correctly answers the question.

Question: {question}
0: {option_0}
1: {option_1}
2: {option_2}

Reply with the index (0, 1, or 2) of the correct option."""


class VLMSelector(Protocol):
    name: str

    def predict(
        self, record_id: str, image_path: Path, question: str, options: tuple[str, str, str]
    ) -> Prediction: ...


class OllamaJointMCQSelector:
    name = "ollama-joint-mcq"

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434",
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._ollama = OllamaClient(base_url=base_url, client=client)

    def predict(
        self, record_id: str, image_path: Path, question: str, options: tuple[str, str, str]
    ) -> Prediction:
        prompt = _PROMPT.format(
            question=question, option_0=options[0], option_1=options[1], option_2=options[2]
        )
        try:
            content = self._ollama.chat(
                self._model, prompt, image_path=image_path, json_schema=_SELECT_SCHEMA
            )
            index = int(json.loads(content)["answer_index"])
            if index not in (0, 1, 2):
                raise ValueError(f"answer_index out of range: {index}")
        except Exception as exc:
            return Prediction(record_id, FALLBACK_INDEX, confidence=None, raw=f"error: {exc}")
        return Prediction(record_id, index, confidence=None, raw=content)
