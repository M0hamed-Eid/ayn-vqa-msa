"""Thin HTTP client for a local Ollama server (default
`http://localhost:11434`).

No official SDK dependency needed -- Ollama's REST `/api/chat` endpoint is
simple enough that one `httpx` call (already a project dependency, from
M2's Fanar/OpenAI clients) covers everything the parse and select-VLM
stages need: a chat completion with an optional image and an optional
JSON-schema-constrained response shape. One client, shared by both
stages, so there is exactly one place that knows the request/response
shape of a local Ollama call.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


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
    ) -> str:
        """Returns the assistant message's raw text content -- a JSON
        string matching `json_schema` when one was given, plain text
        otherwise. Callers own parsing/validating that content; this
        client only knows how to talk to Ollama, not what any particular
        stage expects back.
        """
        import httpx

        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_path is not None:
            message["images"] = [base64.b64encode(image_path.read_bytes()).decode()]

        payload: dict[str, Any] = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        post = (self._client or httpx).post
        response = post(f"{self._base_url}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        content: str = response.json()["message"]["content"]
        return content
