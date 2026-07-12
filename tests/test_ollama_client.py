"""Tests for ollama_client.py using httpx.MockTransport -- intercepts at
the real httpx transport layer, so these verify the actual request our
code sends (URL, body shape) rather than our own assumptions about
httpx's API. No live Ollama server involved.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

from ayn_vqa.ollama_client import ChatMessage, OllamaClient


def test_chat_sends_text_only_request_and_returns_content() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hello"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(base_url="http://localhost:11434", client=client)

    result = ollama.chat("qwen2.5vl:7b", "say hi")

    assert result == "hello"
    assert captured["url"] == "http://localhost:11434/api/chat"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen2.5vl:7b"
    assert body["messages"][0]["content"] == "say hi"
    assert "images" not in body["messages"][0]
    assert "format" not in body


def test_chat_includes_base64_image_when_given(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake image bytes")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": "0"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(client=client)

    ollama.chat("model", "what's in this image?", image_path=image_path)

    body = captured["body"]
    assert isinstance(body, dict)
    expected_b64 = base64.b64encode(b"fake image bytes").decode()
    assert body["messages"][0]["images"] == [expected_b64]


def test_chat_includes_json_schema_format_when_given() -> None:
    captured: dict[str, object] = {}
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": "{}"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(client=client)

    ollama.chat("model", "prompt", json_schema=schema)

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["format"] == schema


def test_chat_omits_num_ctx_by_default() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(client=client)

    ollama.chat("model", "prompt")

    body = captured["body"]
    assert isinstance(body, dict)
    assert "num_ctx" not in body["options"]


def test_chat_sends_num_ctx_when_given() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(client=client)

    ollama.chat("model", "prompt", num_ctx=16384)

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["options"]["num_ctx"] == 16384


def test_chat_messages_sends_a_multi_turn_conversation(tmp_path: Path) -> None:
    exemplar_image = tmp_path / "exemplar.jpg"
    exemplar_image.write_bytes(b"exemplar bytes")
    query_image = tmp_path / "query.jpg"
    query_image.write_bytes(b"query bytes")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": "2"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(client=client)

    messages = [
        ChatMessage(role="user", content="example question", image_path=exemplar_image),
        ChatMessage(role="assistant", content="example answer"),
        ChatMessage(role="user", content="real question", image_path=query_image),
    ]
    result = ollama.chat_messages("model", messages)

    assert result == "2"
    body = captured["body"]
    assert isinstance(body, dict)
    assert len(body["messages"]) == 3
    assert body["messages"][0]["content"] == "example question"
    assert "images" in body["messages"][0]
    assert body["messages"][1]["role"] == "assistant"
    assert "images" not in body["messages"][1]
    assert body["messages"][2]["content"] == "real question"
    assert "images" in body["messages"][2]
