from pathlib import Path

from ayn_vqa.audit.audio_stats import probe_audio


def test_probe_audio_reads_duration_rate_and_channels(mini_dataset: Path) -> None:
    stat = probe_audio("id01", mini_dataset / "audio" / "msa" / "id01.wav")

    assert stat.ok
    assert stat.sample_rate == 16000
    assert stat.channels == 2  # id01 is the stereo fixture, see conftest.mini_dataset
    assert stat.duration_sec is not None
    assert abs(stat.duration_sec - 2.0) < 0.01  # duration_sec=1.0 + i for i=1


def test_probe_audio_missing_file_reports_error_not_exception(tmp_path: Path) -> None:
    stat = probe_audio("ghost", tmp_path / "nope.wav")

    assert not stat.ok
    assert stat.error == "file not found"
    assert stat.duration_sec is None


def test_probe_audio_corrupt_file_reports_error_not_exception(tmp_path: Path) -> None:
    bad_path = tmp_path / "corrupt.wav"
    bad_path.write_bytes(b"this is not a real wav file")

    stat = probe_audio("bad", bad_path)

    assert not stat.ok
    assert stat.error is not None
