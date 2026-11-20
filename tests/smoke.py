"""Smoke test: imports + pre/post-processing shapes; add --full to run the real model.

python -m tests.smoke
python -m tests.smoke --full
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

import lib.video_inpainting as video_inpainting
import run.process_one_video  # noqa: F401 (import check)
from lib.masklets import compose_masklets, free_model, get_masklet_model, make_masklets
from lib.video_inpainting import (
    _composite, _prepare, dilate_masks, _stage_1_long_side, _window_frames, _windows, get_video_model, inpaint_video,
)
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


def test_windows():
    for n, window, overlap in [(98, 97, 25), (97, 97, 25), (25, 97, 25), (300, 97, 25), (40, 17, 9)]:
        spans = _windows(n, window, overlap)
        assert spans[0][0] == 0 and spans[-1][1] == n
        assert all(stop - start <= window for start, stop in spans)
        assert all(prev[1] - cur[0] == overlap for prev, cur in zip(spans, spans[1:]))
    assert _windows(25, 97, 25) == [(0, 25)] and _windows(98, 97, 25) == [(0, 97), (72, 98)]
    for window, overlap in [(96, 25), (97, 24), (25, 25), (33, 33)]:
        try:
            _windows(100, window, overlap)
        except ValueError:
            continue
        raise AssertionError(f"window={window} overlap={overlap} should raise")
    assert _window_frames(448, 768) == 97 and _window_frames(128, 256) == 121


def _identity_generate(frames, gen_masks, model, encoded_prompt, seed, fps, context=None):
    """Stands in for LTX: returns the input, with the overlap pinned to `context` like the real model."""
    stage_1 = frames.copy()
    if context is not None:
        stage_1[: len(context[0])] = context[0]
    stage_2 = stage_1.repeat(2, axis=1).repeat(2, axis=2)
    if context is not None:
        stage_2[: len(context[1])] = context[1]
    return stage_1, stage_2


def test_stitch():
    """Windows with an identity model must reproduce the input exactly (spans, colours, crossfade)."""
    video = np.zeros((40, 64, 128, 3), np.uint8)
    video[...] = np.arange(40, dtype=np.uint8)[:, None, None, None] * np.array([1, 3, 5], np.uint8)
    real = video_inpainting._generate, video_inpainting._encode_prompt
    video_inpainting._generate, video_inpainting._encode_prompt = _identity_generate, lambda *_: None
    kwargs = dict(long_side=128, dilate_px=0, feather_px=0, window=17, overlap=9)
    masks = np.full(video.shape[:3], 255, np.uint8)
    try:
        out = inpaint_video(video, masks, {"pipe": None}, **kwargs)
        first = inpaint_video(video, masks, {"pipe": None}, first_window_only=True, **kwargs)
    finally:
        video_inpainting._generate, video_inpainting._encode_prompt = real
    assert (out == video).all()
    assert (first == video[:17]).all()


def test_dilate():
    masks = np.zeros((2, 64, 64), np.uint8)
    masks[:, 32, 32] = 255
    grown = dilate_masks(masks, 5)
    assert grown.dtype == np.uint8 and grown[:, 32, 37].all() and not grown[:, 32, 38].any()


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


def test_full_model(model):
    video, masks = synthetic_clip()
    out = inpaint_video(video, masks, model, long_side=256)
    assert out.shape == video.shape and out.dtype == np.uint8
    assert (out[:, :10] == video[:, :10]).all(), "pixels far from the mask must be untouched"
    assert (out[masks > 0] != video[masks > 0]).any(), "masked pixels must be regenerated"


def test_full_windows(model):
    video, masks = synthetic_clip(num_frames=40)
    out = inpaint_video(video, masks, model, long_side=256, window=17, overlap=9)
    assert out.shape == video.shape and out.dtype == np.uint8
    assert (out[:, :10] == video[:, :10]).all(), "pixels far from the mask must be untouched"


if __name__ == "__main__":
    test_shapes()
    test_windows()
    test_stitch()
    test_dilate()
    test_compose()
    test_mask_file_round_trip()
    if "--full" in sys.argv:
        test_full_masklets()
        model = get_video_model()
        test_full_model(model)
        test_full_windows(model)
    print("smoke ok")
