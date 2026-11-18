"""Inpaint the masked region of one video.

python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out out.mp4
"""
import argparse

from lib.video_inpainting import get_video_model, inpaint_video
from lib.video_io import read_masks, read_video, write_video


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True, help="mask video; white = region to inpaint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=None, help="describe only the masked region")
    parser.add_argument("--max-frames", type=int, default=97)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    video, fps = read_video(args.video, args.max_frames)
    masks = read_masks(args.masks, args.max_frames)
    model = get_video_model()
    inpainted = inpaint_video(
        video, masks, model, prompt=args.prompt, seed=args.seed, long_side=args.long_side, fps=fps
    )
    write_video(args.out, inpainted, fps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
