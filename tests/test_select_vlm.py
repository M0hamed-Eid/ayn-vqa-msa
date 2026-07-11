from __future__ import annotations

import json
from pathlib import Path

import httpx

from ayn_vqa.stages.select_vlm import FALLBACK_INDEX, OllamaJointMCQSelector


def _client_returning(payload: dict[str, int]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_select_returns_predicted_index(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    client = _client_returning({"answer_index": 2})
    selector = OllamaJointMCQSelector(client=client)

    result = selector.predict("id0", image_path, "question?", ("a", "b", "c"))

    assert result.pred == 2
    assert result.record_id == "id0"


def test_select_falls_back_on_out_of_range_index(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    client = _client_returning({"answer_index": 7})
    selector = OllamaJointMCQSelector(client=client)

    result = selector.predict("id0", image_path, "question?", ("a", "b", "c"))

    assert result.pred == FALLBACK_INDEX
    assert "error" in (result.raw or "")


def test_select_falls_back_on_http_error(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    selector = OllamaJointMCQSelector(client=client)

    result = selector.predict("id0", image_path, "question?", ("a", "b", "c"))

    assert result.pred == FALLBACK_INDEX
    assert "error" in (result.raw or "")


def test_select_sends_image_and_options_in_prompt(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": json.dumps({"answer_index": 0})}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    selector = OllamaJointMCQSelector(client=client)

    selector.predict("id0", image_path, "what is it?", ("cat", "dog", "bird"))

    body = captured["body"]
    assert isinstance(body, dict)
    message = body["messages"][0]
    assert "images" in message
    assert "cat" in message["content"]
    assert "dog" in message["content"]
    assert "bird" in message["content"]
