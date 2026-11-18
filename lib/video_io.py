"""Read & write videos as cv2 arrays: (T, H, W, 3) uint8 BGR."""
import cv2
import numpy as np

DEFAULT_FPS = 24.0


def read_video(path, max_frames=None):
    """Returns (frames, fps); fps falls back to 24 when the container reports nonsense."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else DEFAULT_FPS
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
    height, width = video.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"could not open {path} for writing")
    for frame in video:
        writer.write(frame)
    writer.release()
