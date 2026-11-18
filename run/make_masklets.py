"""Make a masklet video from text prompts, optionally composed with existing mask videos.

python -m run.make_masklets --video in.mp4 --prompt "dog" --out dog.mp4
python -m run.make_masklets --video in.mp4 --prompt "person" --masks face.mp4 --compose subtract --out body.mp4
"""
import argparse

from lib.masklets import COMPOSE_OPS, compose_masklets, free_model, get_masklet_model, make_masklets
from lib.video_io import read_masks, read_video, write_video


def add_masklet_args(parser, prompt_flag="--prompt"):
    parser.add_argument("--masks", nargs="*", default=[], help="mask videos; white = masked")
    parser.add_argument(prompt_flag, dest="masklet_prompts", action="append", default=[],
                        help="SAM3 text prompt for a region to mask, e.g. 'dog'; repeat for several")
    parser.add_argument("--compose", choices=COMPOSE_OPS, default="union",
                        help="how to combine masklets; subtract = first minus the rest (--masks come first)")


def collect_masks(video, mask_paths, prompts, op, max_frames=None):
    """Read mask videos, make prompt masklets, and compose all of them (mask files first)."""
    masklets = [read_masks(path, max_frames) for path in mask_paths]
    if prompts:
        model = get_masklet_model()
        by_prompt = make_masklets(video, prompts, model)
        free_model(model)
        masklets += [by_prompt[prompt] for prompt in prompts]
    return compose_masklets(masklets, op)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="mask video; white = masked")
    parser.add_argument("--max-frames", type=int, default=97)
    add_masklet_args(parser)
    args = parser.parse_args()
    if not args.masks and not args.masklet_prompts:
        parser.error("give at least one --prompt or --masks")
    return args


def main():
    args = parse_args()
    video, fps = read_video(args.video, args.max_frames)
    masks = collect_masks(video, args.masks, args.masklet_prompts, args.compose, args.max_frames)
    write_video(args.out, masks[..., None].repeat(3, axis=-1), fps)
    print(f"wrote {args.out} ({(masks > 0).mean():.1%} of pixels masked)")


if __name__ == "__main__":
    main()
