"""Read & write videos as cv2 arrays: (T, H, W, 3) uint8 BGR."""
import math
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 24.0
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


def check_video(video):
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8 or len(video) == 0:
        raise ValueError(f"video must be (T>0, H, W, 3) uint8 BGR, got {video.shape} {video.dtype}")


def read_video(path, max_frames=None):
    """Returns (frames, fps); fps falls back to 24 when the container reports none."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 and math.isfinite(fps) else DEFAULT_FPS
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"no frames could be read from {path}")
    return np.stack(frames), fps


def read_masks(path, max_frames=None):
    """Read a mask video (white = masked) as (T, H, W) uint8 in {0, 255}."""
    frames, _ = read_video(path, max_frames)
    return ((frames.max(axis=-1) > 127) * 255).astype(np.uint8)


def write_video(path, video, fps):
    """Write BGR frames, or (T, H, W) masks as greyscale; odd sizes are edge-padded to even for mp4v."""
    height, width = video.shape[1:3]
    pad_h, pad_w = height % 2, width % 2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + pad_w, height + pad_h))
    if not writer.isOpened():
        raise ValueError(f"could not open {path} for writing")
    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
        writer.write(cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE))
    writer.release()


def check_output_path(path, inputs=()):
    """Fail fast, before any model runs, on an unwritable or input-clobbering output path."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"{path} must end in one of {VIDEO_SUFFIXES}")
    if not path.parent.is_dir():
        raise ValueError(f"{path.parent} does not exist")
    if any(path.resolve() == Path(p).resolve() for p in inputs):
        raise ValueError(f"{path} would overwrite an input")
