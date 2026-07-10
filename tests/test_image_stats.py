from pathlib import Path

from PIL import Image

from ayn_vqa.audit.image_stats import probe_image


def test_probe_image_reads_resolution_and_format(mini_dataset: Path) -> None:
    stat = probe_image("id00", mini_dataset / "images" / "id00.jpg")

    assert stat.ok
    assert stat.width == 64
    assert stat.height == 48
    assert stat.format == "JPEG"
    assert stat.mode == "RGB"
    assert stat.is_animated is False
    assert stat.n_frames == 1
    assert stat.megapixels is not None


def test_probe_image_missing_file_reports_error_not_exception(tmp_path: Path) -> None:
    stat = probe_image("ghost", tmp_path / "nope.jpg")

    assert not stat.ok
    assert stat.error == "file not found"
    assert stat.width is None


def test_probe_image_corrupt_file_reports_error_not_exception(tmp_path: Path) -> None:
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"not actually an image")

    stat = probe_image("bad", bad_path)

    assert not stat.ok
    assert stat.error is not None


def test_probe_image_detects_animated_gif(tmp_path: Path) -> None:
    gif_path = tmp_path / "anim.gif"
    frame_a = Image.new("RGB", (10, 10), (255, 0, 0))
    frame_b = Image.new("RGB", (10, 10), (0, 255, 0))
    frame_a.save(gif_path, save_all=True, append_images=[frame_b])

    stat = probe_image("anim", gif_path)

    assert stat.ok
    assert stat.format == "GIF"
    assert stat.is_animated is True
    assert stat.n_frames == 2
