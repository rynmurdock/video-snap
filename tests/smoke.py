"""Smoke test: imports + pre/post-processing shapes; add --full to run the real model.

python -m tests.smoke
python -m tests.smoke --full
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

import run.process_one_video  # noqa: F401 (import check)
from lib.masklets import compose_masklets, free_model, get_masklet_model, make_masklets
from lib.video_inpainting import _composite, _prepare, _stage_1_long_side, get_video_model, inpaint_video
from lib.video_io import write_video
from run.make_masklets import collect_masks


def synthetic_clip(num_frames=12, height=180, width=320):
    rng = np.random.default_rng(0)
    video = rng.integers(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
    masks = np.zeros((num_frames, height, width), dtype=np.uint8)
    masks[:, 60:120, 100:220] = 255
    return video, masks


def test_shapes():
    video, masks = synthetic_clip()
    frames, gen_masks = _prepare(video, masks, long_side=256)
    assert frames.shape == (17, 128, 256, 3) and gen_masks.shape == (17, 128, 256)
    assert [_stage_1_long_side(s, 768) for s in ((180, 320), (720, 1280), (2160, 3840))] == [256, 640, 768]
    generated = np.zeros((len(video), 256, 512, 3), dtype=np.uint8)
    out = _composite(video, masks, generated, feather_px=0)
    assert out.shape == video.shape and out.dtype == np.uint8
    assert (out[masks > 0] == 0).all() and (out[masks == 0] == video[masks == 0]).all()


def test_compose():
    a, b = np.zeros((2, 4, 4), np.uint8), np.zeros((2, 4, 4), np.uint8)
    a[:, :2], b[:, :, :2] = 255, 255
    assert (compose_masklets([a, b], "union") > 0).sum() == 2 * 12
    assert (compose_masklets([a, b], "intersect") > 0).sum() == 2 * 4
    assert (compose_masklets([a, b], "subtract") > 0).sum() == 2 * 4
    assert compose_masklets([a], "subtract").dtype == np.uint8


def test_mask_file_round_trip():
    video, masks = synthetic_clip(height=181, width=321)  # odd sizes get padded by write_video
    with tempfile.TemporaryDirectory() as tmp:
        write_video(Path(tmp) / "masks.mp4", masks, 24)
        read_back = collect_masks(video, [Path(tmp) / "masks.mp4"], [], "union")
    assert (read_back == masks).all()


def test_full_masklets():
    video, _ = synthetic_clip()
    model = get_masklet_model()
    masklets = make_masklets(video, ["dog", "car"], model)
    free_model(model)
    assert set(masklets) == {"dog", "car"}
    assert all(m.shape == video.shape[:3] and m.dtype == np.uint8 for m in masklets.values())


def test_full_model():
    video, masks = synthetic_clip()
    out = inpaint_video(video, masks, get_video_model(), long_side=256)
    assert out.shape == video.shape and out.dtype == np.uint8
    assert (out[:, :10] == video[:, :10]).all(), "pixels far from the mask must be untouched"
    assert (out[masks > 0] != video[masks > 0]).any(), "masked pixels must be regenerated"


if __name__ == "__main__":
    test_shapes()
    test_compose()
    test_mask_file_round_trip()
    if "--full" in sys.argv:
        test_full_masklets()
        test_full_model()
    print("smoke ok")
