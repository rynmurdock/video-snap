"""Smoke test: imports + pre/post-processing shapes; add --full to run the real model.

python -m tests.smoke
python -m tests.smoke --full
"""
import sys

import numpy as np

import run.process_one_video  # noqa: F401 (import check)
from lib.video_inpainting import _composite, _prepare, get_video_model, inpaint_video


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
    generated = np.zeros((len(video), 256, 512, 3), dtype=np.uint8)
    out = _composite(video, masks, generated, feather_px=0)
    assert out.shape == video.shape and out.dtype == np.uint8
    assert (out[masks > 0] == 0).all() and (out[masks == 0] == video[masks == 0]).all()


def test_full_model():
    video, masks = synthetic_clip()
    out = inpaint_video(video, masks, get_video_model(), long_side=256)
    assert out.shape == video.shape and out.dtype == np.uint8
    assert (out[:, :40] == video[:, :40]).all(), "pixels far from the mask must be untouched"
    assert (out[masks > 0] != video[masks > 0]).any(), "masked pixels must be regenerated"


if __name__ == "__main__":
    test_shapes()
    if "--full" in sys.argv:
        test_full_model()
    print("smoke ok")
