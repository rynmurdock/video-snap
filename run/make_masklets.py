"""Make a masklet video from text prompts, optionally composed with existing mask videos.

python -m run.make_masklets --video in.mp4 --masklet-prompt "dog" --out dog.mp4
python -m run.make_masklets --video in.mp4 --masks person.mp4 --masklet-prompt "face" --compose subtract --out body.mp4
"""
import argparse

from lib.masklets import COMPOSE_OPS, compose_masklets, free_model, get_masklet_model, make_masklets
from lib.video_io import check_output_path, read_masks, read_video, write_video


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def add_masklet_args(parser):
    parser.add_argument("--masks", nargs="+", action="extend", default=[], help="mask videos; white = masked")
    parser.add_argument("--masklet-prompt", dest="masklet_prompts", action="append", default=[],
                        help="SAM3 text prompt for a region to mask, e.g. 'dog'; repeat for several")
    parser.add_argument("--compose", choices=COMPOSE_OPS, default="union",
                        help="how to combine masklets; subtract = first minus the rest (--masks come first)")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="default: the whole video")


def check_masklet_args(parser, args, outputs):
    """Fail on bad arguments before any model loads."""
    if not args.masks and not args.masklet_prompts:
        parser.error("give at least one --masks or --masklet-prompt")
    try:
        for out in outputs:
            check_output_path(out, inputs=[args.video, *args.masks])
    except ValueError as error:
        parser.error(str(error))


def collect_masks(video, mask_paths, prompts, op):
    """Read mask videos, make prompt masklets, and compose all of them (mask files first)."""
    masklets = [_read_matching_masks(path, video) for path in mask_paths]
    # SAM3's CLIP tokenizer lowercases, so prompts differing only in case are the same prompt.
    prompts = list({prompt.strip().lower(): prompt.strip() for prompt in prompts}.values())
    if prompts:
        model = get_masklet_model()
        by_prompt = make_masklets(video, prompts, model)
        free_model(model)
        masklets += [by_prompt[prompt] for prompt in prompts]
    return compose_masklets(masklets, op)


def _read_matching_masks(path, video):
    """Read a mask video trimmed to the video's frames; drops write_video's 1px odd-size padding."""
    num_frames, height, width = video.shape[:3]
    masks = read_masks(path, num_frames)
    if len(masks) != num_frames:
        raise ValueError(f"{path} has {len(masks)} frames, video has {num_frames}")
    if masks.shape[1:] not in {(height, width), (height + height % 2, width + width % 2)}:
        raise ValueError(f"{path} is {masks.shape[2]}x{masks.shape[1]}, video is {width}x{height}")
    return masks[:, :height, :width]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="mask video; white = masked")
    add_masklet_args(parser)
    args = parser.parse_args()
    check_masklet_args(parser, args, outputs=[args.out])
    return args


def main():
    args = parse_args()
    video, fps = read_video(args.video, args.max_frames)
    masks = collect_masks(video, args.masks, args.masklet_prompts, args.compose)
    write_video(args.out, masks, fps)
    print(f"wrote {args.out} ({(masks > 0).mean():.1%} of pixels masked)")


if __name__ == "__main__":
    main()
