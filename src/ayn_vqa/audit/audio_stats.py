"""Audio statistics: duration, sample rate, channel count, subtype.

Uses `soundfile.info`, which reads only the WAV header via libsndfile --
not the full sample buffer -- so probing thousands of files takes seconds,
not minutes. This also works uniformly across PCM and float WAV subtypes,
unlike the stdlib `wave` module, which understands PCM only and raises on
anything else (a real risk here: TTS/voice-cloning pipelines sometimes
emit float32 WAV).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class AudioStat:
    """Probe result for one audio file. `error` is set (and every numeric
    field is `None`) when the file is missing or libsndfile can't open it --
    the caller decides what to do with that, this dataclass never raises.
    """

    record_id: str
    path: Path
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    subtype: str | None
    frames: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def probe_audio(record_id: str, path: Path) -> AudioStat:
    if not path.is_file():
        return AudioStat(record_id, path, None, None, None, None, None, error="file not found")
    try:
        info = sf.info(str(path))
    except Exception as exc:
        # Any libsndfile failure (corrupt header, unsupported codec, zero-byte
        # file) must become a recorded finding, not a crashed audit run.
        return AudioStat(record_id, path, None, None, None, None, None, error=str(exc))

    duration = info.frames / info.samplerate if info.samplerate else None
    return AudioStat(
        record_id=record_id,
        path=path,
        duration_sec=duration,
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype,
        frames=info.frames,
        error=None,
    )


def probe_many(items: Iterable[tuple[str, Path]]) -> list[AudioStat]:
    return [probe_audio(record_id, path) for record_id, path in items]
