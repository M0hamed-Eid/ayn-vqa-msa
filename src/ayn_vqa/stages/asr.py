"""ASR stage: audio -> transcript text.

`ASRBackend` is the swappable interface every transcription system
implements, matching `stages/select.py`'s `Protocol` pattern. Only
`WhisperLocalASR` is actually exercised in this project so far: no API
key, no per-call cost, runs entirely offline. `FanarAuraASR` and
`OpenAITranscribeASR` are complete, ready-to-use clients gated behind
their own API key -- add one to `.env` and they work with zero code
changes; see docs/M2_ASR_BENCH.md for exactly what has and hasn't been
run against a live API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Transcript:
    """One backend's transcription of one record. `error` is set (`text`
    `None`) on failure -- a bad file or a flaky API call becomes a
    recorded finding, not a crashed bench run.
    """

    record_id: str
    text: str | None
    backend_name: str
    model_name: str
    language: str | None
    latency_sec: float | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


class ASRBackend(Protocol):
    name: str

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript: ...


class _WhisperModelLike(Protocol):
    """The one method we actually call on a `faster_whisper.WhisperModel`.
    Naming this lets tests inject a fake here instead of downloading real
    model weights just to exercise `WhisperLocalASR.transcribe`'s own
    logic (latency timing, joining segments, error handling).
    """

    def transcribe(self, audio: str, *, language: str, beam_size: int) -> tuple[Any, Any]: ...


class WhisperLocalASR:
    """Local, offline transcription via `faster-whisper` (a CTranslate2
    reimplementation of Whisper) -- much faster than plain `openai-whisper`
    on CPU, which matters here since this machine has no confirmed GPU.

    Model weights download from Hugging Face on first use per `model_size`
    and are cached under the user's HF cache afterward; no API key, no
    per-call cost.
    """

    name = "whisper-local"

    def __init__(
        self,
        model_size: str = "small",
        language: str = "ar",
        device: str = "cpu",
        compute_type: str = "int8",
        model: _WhisperModelLike | None = None,
    ) -> None:
        self._model_size = model_size
        self._language = language
        if model is not None:
            self._model = model
        else:
            from faster_whisper import WhisperModel  # heavy import; deferred to construction

            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript:
        start = time.monotonic()
        try:
            segments, _info = self._model.transcribe(
                str(audio_path), language=self._language, beam_size=5
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            return Transcript(
                record_id, None, self.name, self._model_size, self._language, None, str(exc)
            )
        latency = time.monotonic() - start
        return Transcript(
            record_id, text, self.name, self._model_size, self._language, latency, None
        )


class FanarAuraASR:
    """Fanar `Aura-STT` via the QCRI Fanar API -- the Arabic-specialized
    option the project's plan doc flags as plausibly the strongest MSA
    transcriber available. Request/response shape copied faithfully from
    the organizers' own baseline (`ImageEval2026-tasks/task1/baselines/
    baseline_task1a_fanar_cascade_colab.ipynb`), so a transcript from this
    backend is directly comparable to their published cascade baseline.

    **Not yet exercised against the live API** in this project (no key
    configured) -- ready to use the moment `AYNVQA_FANAR_API_KEY` is set.
    """

    name = "fanar-aura-stt"

    def __init__(
        self,
        api_key: str | None,
        model: str = "Fanar-Aura-STT-1",
        base_url: str = "https://api.fanar.qa/v1",
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "FanarAuraASR requires an API key -- set AYNVQA_FANAR_API_KEY in .env "
                "(request one at https://api.fanar.qa/request)."
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client  # injected httpx.Client, for tests; else built per call

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript:
        import httpx

        start = time.monotonic()
        try:
            with audio_path.open("rb") as f:
                post = (self._client or httpx).post
                response = post(
                    f"{self._base_url}/audio/transcriptions",
                    headers=self._headers,
                    files={"file": f},
                    data={"model": self._model, "format": "text"},
                    timeout=120,
                )
            response.raise_for_status()
            try:
                text = response.json().get("text", response.text).strip()
            except ValueError:
                text = response.text.strip()
        except Exception as exc:
            return Transcript(record_id, None, self.name, self._model, "msa", None, str(exc))
        latency = time.monotonic() - start
        return Transcript(record_id, text, self.name, self._model, "msa", latency, None)


class OpenAITranscribeASR:
    """OpenAI's hosted transcription API (`gpt-4o-transcribe` by default).

    **Not yet exercised against the live API** in this project (no key
    configured) -- ready to use the moment `AYNVQA_OPENAI_API_KEY` is set.
    """

    name = "openai-transcribe"

    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-4o-transcribe",
        base_url: str = "https://api.openai.com/v1",
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "OpenAITranscribeASR requires an API key -- set AYNVQA_OPENAI_API_KEY in .env."
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client

    def transcribe(self, record_id: str, audio_path: Path) -> Transcript:
        import httpx

        start = time.monotonic()
        try:
            with audio_path.open("rb") as f:
                post = (self._client or httpx).post
                response = post(
                    f"{self._base_url}/audio/transcriptions",
                    headers=self._headers,
                    files={"file": f},
                    data={"model": self._model, "response_format": "json"},
                    timeout=120,
                )
            response.raise_for_status()
            text = response.json()["text"].strip()
        except Exception as exc:
            return Transcript(record_id, None, self.name, self._model, None, None, str(exc))
        latency = time.monotonic() - start
        return Transcript(record_id, text, self.name, self._model, None, latency, None)
