"""Thin HTTP client for a local Ollama server (default
`http://localhost:11434`).

No official SDK dependency needed -- Ollama's REST `/api/chat` endpoint is
simple enough that one `httpx` call (already a project dependency, from
M2's Fanar/OpenAI clients) covers everything the parse and select-VLM
stages need: a chat completion with an optional image and an optional
JSON-schema-constrained response shape. One client, shared by every
stage, so there is exactly one place that knows the request/response
shape of a local Ollama call.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a multi-turn conversation (M5's few-shot exemplars:
    each exemplar is a user turn with its image/question/options followed
    by an assistant turn stating the correct answer, ending with the real
    query as the final user turn)."""

    role: str  # "user" or "assistant"
    content: str
    image_path: Path | None = None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", client: Any | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client  # injected httpx.Client, for tests; else the httpx module

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        image_path: Path | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        num_ctx: int | None = None,
    ) -> str:
        """Single-turn convenience wrapper around `chat_messages` -- every
        M0-M4 caller's shape, unchanged."""
        return self.chat_messages(
            model,
            [ChatMessage(role="user", content=prompt, image_path=image_path)],
            json_schema=json_schema,
            temperature=temperature,
            timeout=timeout,
            num_ctx=num_ctx,
        )

    def chat_messages(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        num_ctx: int | None = None,
    ) -> str:
        """Returns the assistant message's raw text content -- a JSON
        string matching `json_schema` when one was given, plain text
        otherwise. Callers own parsing/validating that content; this
        client only knows how to talk to Ollama, not what any particular
        stage expects back.

        `num_ctx` matters for multi-image few-shot prompts specifically --
        Ollama's own default context window (4096 tokens) is too small for
        more than one image; left `None` (Ollama's default) for every
        caller that doesn't need it, rather than raising it globally.
        """
        import httpx

        built_messages: list[dict[str, Any]] = []
        for message in messages:
            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.image_path is not None:
                entry["images"] = [base64.b64encode(message.image_path.read_bytes()).decode()]
            built_messages.append(entry)

        options: dict[str, Any] = {"temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload: dict[str, Any] = {
            "model": model,
            "messages": built_messages,
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            payload["format"] = json_schema

        post = (self._client or httpx).post
        response = post(f"{self._base_url}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        content: str = response.json()["message"]["content"]
        return content
