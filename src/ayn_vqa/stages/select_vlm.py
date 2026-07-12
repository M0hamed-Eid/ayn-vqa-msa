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

from ayn_vqa.ollama_client import ChatMessage, OllamaClient
from ayn_vqa.stages.retrieve import Exemplar
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

# Chain-of-thought variant (M5). Same image/question/options, same
# constrained-decoding guarantee -- `answer_index` is still schema-locked
# to {0,1,2} -- but the schema puts `visible_details`/`reasoning` *before*
# `answer_index`, so constrained decoding generates them first and the
# final index is conditioned on that reasoning rather than produced cold.
# A cheap sanity pilot on the 43 dev items the frozen M4 baseline got
# wrong for genuine visual-reasoning reasons (not pipeline artifacts)
# recovered 13/43 this way, zero retrieval/exemplar infrastructure
# involved -- see docs/M5_FEWSHOT_RETRIEVAL.md for the full-scale result
# this prompt was validated against before being adopted or not.
_SELECT_SCHEMA_COT = {
    "type": "object",
    "properties": {
        "visible_details": {"type": "string"},
        "reasoning": {"type": "string"},
        "answer_index": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["visible_details", "reasoning", "answer_index"],
}

_PROMPT_COT = """You are given an image and a multiple-choice question about it, \
in Arabic. Using ONLY what is visible in the image, choose the option that \
correctly answers the question.

Question: {question}
0: {option_0}
1: {option_1}
2: {option_2}

First, describe the specific visible details in the image relevant to this \
question (materials, shapes, colors, text, distinguishing features). Then \
reason step by step about which option those details support. Finally give \
the index (0, 1, or 2) of the correct option."""

# Few-shot exemplar turn (M5). Deliberately simpler than `_PROMPT` -- no
# "reply with the index" instruction, since the answer is supplied
# directly as the following assistant turn, not generated. `_run_pipeline`
# builds one of these per retrieved exemplar; see `stages/retrieve.py`.
_EXEMPLAR_PROMPT = """Example question about a different image, in Arabic:

Question: {question}
0: {option_0}
1: {option_1}
2: {option_2}"""

_DEFAULT_FEWSHOT_NUM_CTX = 16384  # Ollama's own 4096 default 400s on >1 image; see M5 report.


class VLMSelector(Protocol):
    name: str

    def predict(
        self,
        record_id: str,
        image_path: Path,
        question: str,
        options: tuple[str, str, str],
        exemplars: tuple[Exemplar, ...] = (),
    ) -> Prediction: ...


class OllamaJointMCQSelector:
    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434",
        client: Any | None = None,
        use_cot: bool = False,
        fewshot_num_ctx: int = _DEFAULT_FEWSHOT_NUM_CTX,
    ) -> None:
        self._model = model
        self._ollama = OllamaClient(base_url=base_url, client=client)
        self._use_cot = use_cot
        self._fewshot_num_ctx = fewshot_num_ctx
        # Distinct cache/output key when CoT changes what this stage
        # produces -- the same lesson M4's parser-v2 swap had to learn the
        # hard way: without this, a cot-on and cot-off run would collide on
        # the same select-stage cache file and silently replay each
        # other's predictions on a second run. Few-shot doesn't get a
        # `.name` variant here -- it's a per-call, per-record concern
        # (exemplars come from a separate retrieval stage), so its cache
        # isolation is `run_pipeline.py`'s `effective_selector_config_key`
        # suffix, the same mechanism M4's repair ladder already uses.
        self.name = "ollama-joint-mcq-cot" if use_cot else "ollama-joint-mcq"

    def predict(
        self,
        record_id: str,
        image_path: Path,
        question: str,
        options: tuple[str, str, str],
        exemplars: tuple[Exemplar, ...] = (),
    ) -> Prediction:
        template = _PROMPT_COT if self._use_cot else _PROMPT
        schema = _SELECT_SCHEMA_COT if self._use_cot else _SELECT_SCHEMA
        final_prompt = template.format(
            question=question, option_0=options[0], option_1=options[1], option_2=options[2]
        )

        messages: list[ChatMessage] = []
        for exemplar in exemplars:
            exemplar_prompt = _EXEMPLAR_PROMPT.format(
                question=exemplar.question,
                option_0=exemplar.options[0],
                option_1=exemplar.options[1],
                option_2=exemplar.options[2],
            )
            messages.append(
                ChatMessage(role="user", content=exemplar_prompt, image_path=exemplar.image_path)
            )
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"The correct answer is: {exemplar.correct_option_text}",
                )
            )
        messages.append(ChatMessage(role="user", content=final_prompt, image_path=image_path))

        try:
            content = self._ollama.chat_messages(
                self._model,
                messages,
                json_schema=schema,
                num_ctx=self._fewshot_num_ctx if exemplars else None,
            )
            index = int(json.loads(content)["answer_index"])
            if index not in (0, 1, 2):
                raise ValueError(f"answer_index out of range: {index}")
        except Exception as exc:
            return Prediction(record_id, FALLBACK_INDEX, confidence=None, raw=f"error: {exc}")
        return Prediction(record_id, index, confidence=None, raw=content)
