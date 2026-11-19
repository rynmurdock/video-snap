"""Text-prompted video masklets with SAM3, plus composition of masklets.

Videos are cv2 arrays: (T, H, W, 3) uint8 BGR. Masklets are (T, H, W) uint8 in {0, 255}.
"""
import gc

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from lib.video_io import check_video

DTYPE = torch.bfloat16
COMPOSE_OPS = ("union", "intersect", "subtract")
MAX_PROMPT_TOKENS = 32


def get_masklet_model(model_path="facebook/sam3"):
    """Load SAM3 for video; `model_path` is the gated SAM3 repo (accept its license first)."""
    model = Sam3VideoModel.from_pretrained(model_path, dtype=DTYPE).to("cuda").eval()
    return {"model": model, "processor": Sam3VideoProcessor.from_pretrained(model_path)}


def free_model(model):
    """Drop any model dict's GPU memory, e.g. SAM3 before loading the inpainting model."""
    model.clear()
    gc.collect()
    torch.cuda.empty_cache()


def make_masklet(video, prompt, model):
    """One masklet covering every instance SAM3 finds for `prompt` (e.g. "dog")."""
    return make_masklets(video, [prompt], model)[prompt]


@torch.inference_mode()
def make_masklets(video, prompts, model):
    """One masklet per prompt from a single tracking pass; returns {prompt: masklet}."""
    check_video(video)
    processor, sam = model["processor"], model["model"]
    _check_prompts(prompts, processor.tokenizer)
    # Frames stay on the CPU: only the tracker state lives on the GPU.
    session = processor.init_video_session(
        video=np.ascontiguousarray(video[..., ::-1]), inference_device="cuda",
        processing_device="cpu", video_storage_device="cpu", dtype=DTYPE,
    )
    processor.add_text_prompt(session, list(prompts))

    masklets = {prompt: np.zeros(video.shape[:3], dtype=np.uint8) for prompt in prompts}
    for outputs in sam.propagate_in_video_iterator(session):
        frame = processor.postprocess_outputs(session, outputs)
        masks, object_ids = frame["masks"], frame["object_ids"].tolist()
        for prompt, prompt_ids in frame["prompt_to_obj_ids"].items():
            rows = [object_ids.index(obj_id) for obj_id in prompt_ids]
            masklets[prompt][outputs.frame_idx] = np.where(masks[rows].any(dim=0).cpu().numpy(), 255, 0)
    return masklets


def compose_masklets(masklets, op="union"):
    """Combine masklets: union / intersect all, or subtract the rest from the first."""
    if op not in COMPOSE_OPS:
        raise ValueError(f"op must be one of {COMPOSE_OPS}, got {op!r}")
    if not masklets:
        raise ValueError("need at least one masklet to compose")
    binary = [np.asarray(masklet) > 0 for masklet in masklets]
    shapes = {masklet.shape for masklet in binary}
    if len(shapes) != 1 or binary[0].ndim != 3:
        raise ValueError(f"masklets must share one (T, H, W) shape, got {sorted(shapes)}")
    if op == "union":
        out = np.logical_or.reduce(binary)
    elif op == "intersect":
        out = np.logical_and.reduce(binary)
    else:
        out = binary[0] & ~np.logical_or.reduce(binary[1:]) if len(binary) > 1 else binary[0]
    return out.astype(np.uint8) * 255


def _check_prompts(prompts, tokenizer):
    """Fail before tracking: SAM3's text encoder has 32 positions & does not truncate."""
    if isinstance(prompts, str):
        raise TypeError("prompts must be a list; use make_masklet for one prompt")
    for prompt in prompts:
        if not prompt.strip():
            raise ValueError("masklet prompts must be non-empty")
        if len(tokenizer(prompt).input_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"masklet prompt is over {MAX_PROMPT_TOKENS} tokens; shorten it: {prompt!r}")
