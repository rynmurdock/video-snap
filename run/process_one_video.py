"""Inpaint the masked region of one video; masks come from mask videos and/or SAM3 prompts.

python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out out.mp4
python -m run.process_one_video --video in.mp4 --masklet-prompt "dog" --prompt "a red fox" --out out.mp4
"""
import argparse

from lib.video_inpainting import get_video_model, inpaint_video
from lib.video_io import read_video, write_video
from run.make_masklets import add_masklet_args, collect_masks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    add_masklet_args(parser, prompt_flag="--masklet-prompt")
    parser.add_argument("--save-masks", default=None, help="also write the composed mask video here")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=None, help="describe only the masked region")
    parser.add_argument("--max-frames", type=int, default=97)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.masks and not args.masklet_prompts:
        parser.error("give at least one --masks or --masklet-prompt")
    return args


def main():
    args = parse_args()
    video, fps = read_video(args.video, args.max_frames)
    # SAM3 is freed inside collect_masks before the inpainting model loads.
    masks = collect_masks(video, args.masks, args.masklet_prompts, args.compose, args.max_frames)
    if args.save_masks:
        write_video(args.save_masks, masks[..., None].repeat(3, axis=-1), fps)
    model = get_video_model()
    inpainted = inpaint_video(
        video, masks, model, prompt=args.prompt, seed=args.seed, long_side=args.long_side, fps=fps
    )
    write_video(args.out, inpainted, fps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
