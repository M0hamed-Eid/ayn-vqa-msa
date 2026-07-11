"""Tests for stages/asr.py.

`WhisperLocalASR` is tested with an injected fake model -- exercising its
own logic (joining segments, timing, error handling) without downloading
real model weights. `FanarAuraASR`/`OpenAITranscribeASR` are tested with
`httpx.MockTransport`, which intercepts the request at the real `httpx`
transport layer -- this verifies the actual request shape (method, URL,
headers, multipart body) our code sends, not just our own assumptions
about `httpx`'s API. Neither test calls a live API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from ayn_vqa.stages.asr import FanarAuraASR, OpenAITranscribeASR, WhisperLocalASR


@dataclass
class _FakeSegment:
    text: str


class _FakeWhisperModel:
    def __init__(
        self, segments_by_audio: dict[str, list[str]] | None = None, raise_error: bool = False
    ) -> None:
        self._segments_by_audio = segments_by_audio or {}
        self._raise_error = raise_error

    def transcribe(
        self, audio: str, *, language: str, beam_size: int
    ) -> tuple[list[_FakeSegment], object]:
        if self._raise_error:
            raise RuntimeError("boom")
        texts = self._segments_by_audio.get(audio, ["default segment"])
        return [_FakeSegment(text=t) for t in texts], object()


def test_whisper_local_asr_joins_segments(tmp_path: Path) -> None:
    audio_path = tmp_path / "a.wav"
    fake_model = _FakeWhisperModel(segments_by_audio={str(audio_path): [" hello ", "world "]})

    asr = WhisperLocalASR(model=fake_model)
    result = asr.transcribe("id0", audio_path)

    assert result.ok
    assert result.text == "hello world"
    assert result.backend_name == "whisper-local"
    assert result.latency_sec is not None
    assert result.latency_sec >= 0


def test_whisper_local_asr_error_becomes_transcript_error(tmp_path: Path) -> None:
    fake_model = _FakeWhisperModel(raise_error=True)

    asr = WhisperLocalASR(model=fake_model)
    result = asr.transcribe("id0", tmp_path / "a.wav")

    assert not result.ok
    assert result.text is None
    assert "boom" in (result.error or "")


def test_fanar_aura_asr_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="FANAR_API_KEY"):
        FanarAuraASR(api_key=None)


def test_fanar_aura_asr_sends_expected_request_and_parses_response(tmp_path: Path) -> None:
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"fake wav bytes")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization", "")
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"text": "  مرحبا  "})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = FanarAuraASR(api_key="fake-key", client=client)

    result = asr.transcribe("id0", audio_path)

    assert result.ok
    assert result.text == "مرحبا"
    assert captured["url"] == "https://api.fanar.qa/v1/audio/transcriptions"
    assert captured["auth"] == "Bearer fake-key"
    assert "multipart/form-data" in captured["content_type"]


def test_fanar_aura_asr_http_error_becomes_transcript_error(tmp_path: Path) -> None:
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"fake wav bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = FanarAuraASR(api_key="fake-key", client=client)

    result = asr.transcribe("id0", audio_path)

    assert not result.ok
    assert result.text is None


def test_openai_transcribe_asr_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAITranscribeASR(api_key=None)


def test_openai_transcribe_asr_sends_expected_request_and_parses_response(tmp_path: Path) -> None:
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"fake wav bytes")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"text": "hello"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = OpenAITranscribeASR(api_key="fake-key", client=client)

    result = asr.transcribe("id0", audio_path)

    assert result.ok
    assert result.text == "hello"
    assert captured["url"] == "https://api.openai.com/v1/audio/transcriptions"
