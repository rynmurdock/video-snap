"""Inpaint the masked region of one video; masks come from mask videos and/or SAM3 prompts.

python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out out.mp4
python -m run.process_one_video --video in.mp4 --masklet-prompt "dog" --prompt "a red fox" --out out.mp4
python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out out.mp4 --distilled  # faster, lower quality
python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out preview.mp4 --first-window  # quick preview
"""
import argparse

from lib.video_inpainting import get_video_model, inpaint_video
from lib.video_io import read_video, write_video
from run.make_masklets import add_masklet_args, check_masklet_args, collect_masks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    add_masklet_args(parser)
    parser.add_argument("--save-masks", default=None, help="also write the composed mask video here")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=None, help="describe only the masked region")
    parser.add_argument("--long-side", type=int, default=512, help="cap on stage-1 size; stage 2 renders at 2x")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distilled", action="store_true", help="use the fast distilled LTX instead of dev + distilled-LoRA refine")
    parser.add_argument("--first-window", action="store_true", help="render only the first window (a short preview)")
    args = parser.parse_args()
    check_masklet_args(parser, args, outputs=[args.out, *([args.save_masks] if args.save_masks else [])])
    return args


def main():
    args = parse_args()
    video, fps = read_video(args.video, args.max_frames)
    # SAM3 is freed inside collect_masks before the inpainting model loads.
    masks = collect_masks(video, args.masks, args.masklet_prompts, args.compose)
    if args.save_masks:
        write_video(args.save_masks, masks, fps)
    if not masks.any():
        raise SystemExit("masks are empty; nothing to inpaint")
    model = get_video_model(distilled=args.distilled)
    inpainted = inpaint_video(
        video, masks, model, prompt=args.prompt, seed=args.seed, long_side=args.long_side, fps=fps,
        first_window_only=args.first_window,
    )
    write_video(args.out, inpainted, fps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
