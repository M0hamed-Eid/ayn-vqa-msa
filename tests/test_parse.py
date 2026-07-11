from __future__ import annotations

import json

import httpx

from ayn_vqa.stages.parse import OllamaTranscriptParser


def _client_returning(payload: dict[str, str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_success() -> None:
    client = _client_returning(
        {"question": "What is this?", "option_0": "a", "option_1": "b", "option_2": "c"}
    )
    parser = OllamaTranscriptParser(client=client)

    result = parser.parse("id0", "some transcript")

    assert result.ok
    assert result.question == "What is this?"
    assert result.options == ("a", "b", "c")
    assert result.error is None


def test_parse_missing_field_becomes_error() -> None:
    client = _client_returning({"question": "q", "option_0": "a", "option_1": "b"})  # no option_2
    parser = OllamaTranscriptParser(client=client)

    result = parser.parse("id0", "transcript")

    assert not result.ok
    assert result.question is None
    assert result.options is None


def test_parse_http_error_becomes_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    parser = OllamaTranscriptParser(client=client)

    result = parser.parse("id0", "transcript")

    assert not result.ok


def test_parse_non_json_content_becomes_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not json at all"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    parser = OllamaTranscriptParser(client=client)

    result = parser.parse("id0", "transcript")

    assert not result.ok
