"""Mask-conditioned video inpainting with the LTX-2.3 In/Outpainting IC-LoRA.

Videos are cv2 arrays: (T, H, W, 3) uint8 BGR. Masks are (T, H, W), nonzero = inpaint.
Follows diffusers' LTX-2 two-stage recipe: IC-LoRA generation from a green-keyed reference, 2x latent upsample,
then a 3-step refine (dev + distilled LoRA by default; the distilled model with `distilled=True`).
Long videos run in overlapping windows; each window is pinned to the previous one's overlap frames.
"""
import contextlib

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2InContextPipeline
from diffusers.models.latent_upscaler.latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
from diffusers.pipelines.ltx2.utils import (
    DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
)
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from lib.video_io import check_video

BASE_REPO = "diffusers/LTX-2.3-Diffusers"
DISTILLED_BASE_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
LORA_FILE = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
LTX_REPO = "Lightricks/LTX-2.3"
DISTILLED_LORA_FILE = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"  # stage-2 refiner on top of dev
UPSAMPLER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

DTYPE = torch.bfloat16
FRAME_STEP = 8  # the VAE compresses time 8x, so frame counts must be 8k + 1
SIZE_STEP = 32
MIN_STAGE_1_LONG_SIDE = 256  # below this LTX output degrades; tiny clips render above source size
GREEN_RGB = (102, 255, 0)  # the IC-LoRA was trained with masked pixels painted this key colour
DEV_STEPS = 30
# diffusers' recommended LTX-2.3 guidance for the dev model (the pipeline's own defaults).
DEV_GUIDANCE = dict(
    guidance_scale=3.0, stg_scale=1.0, modality_scale=3.0, guidance_rescale=0.7, spatio_temporal_guidance_blocks=[28],
    audio_guidance_scale=7.0, audio_stg_scale=1.0, audio_modality_scale=3.0, audio_guidance_rescale=0.7,
)
# Distilled weights run unguided; every audio knob too, since audio CFG > 1 alone switches CFG on.
NO_GUIDANCE = dict(
    guidance_scale=1.0, stg_scale=0.0, modality_scale=1.0, guidance_rescale=0.0,
    audio_guidance_scale=1.0, audio_stg_scale=0.0, audio_modality_scale=1.0, audio_guidance_rescale=0.0,
)
# Stage-2 latent tokens per window: 97 frames (13 latent) of 1080p (28x48 tokens) is the measured-safe 4090 point.
STAGE_2_TOKEN_BUDGET = 13 * 28 * 48
MAX_WINDOW_FRAMES = 121  # measured to fit at 1080p (21.3G); also caps small sources near LTX's usual lengths
OVERLAP_FRAMES = 25  # 4 latent frames, ~1 s of shared context between windows


def get_video_model(model_path="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting", distilled=False):
    """Load the inpainting models; `model_path` is the IC-LoRA repo, `distilled` picks the fast base model."""
    pipe = LTX2InContextPipeline.from_pretrained(DISTILLED_BASE_REPO if distilled else BASE_REPO, torch_dtype=DTYPE)
    # Kept as togglable adapters (not fused): stage 1 uses only the IC-LoRA, stage 2 only the distilled one.
    _load_lora(pipe, model_path, LORA_FILE, "inpaint")
    if not distilled:
        _load_lora(pipe, LTX_REPO, DISTILLED_LORA_FILE, "distilled")
        pipe.set_adapters(["inpaint"], [1.0])
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload(device="cuda")
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=_load_upsampler().to("cuda"))
    # As in diffusers' recipe: stage 2 sigmas are used as given, without the dev scheduler's time shift.
    refine_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
    )
    return {"pipe": pipe, "distilled": distilled, "refine_scheduler": refine_scheduler, "upsample_pipe": upsample_pipe}


def _load_lora(pipe, repo, weight_name, adapter_name):
    """Load a LoRA and check every layer in the file attached; unmatched keys only log a warning."""
    pipe.load_lora_weights(repo, weight_name=weight_name, adapter_name=adapter_name)
    with safe_open(hf_hub_download(repo, weight_name), "pt") as f:
        expected = sum(key.endswith("lora_A.weight") for key in f.keys())
    attached = sum(adapter_name in getattr(m, "lora_A", {}) for m in pipe.transformer.modules())
    if attached != expected:
        raise RuntimeError(f"{weight_name} attached to {attached} of its {expected} layers")


def _load_upsampler():
    """LTX-2.3's 2x latent upsampler (x2 1.1); no diffusers-format release exists, so build it from the file's config."""
    upsampler = LTX2LatentUpsamplerModel(
        in_channels=128, mid_channels=1024, num_blocks_per_stage=4, dims=3,
        spatial_upsample=True, temporal_upsample=False, use_rational_resampler=False,
    )
    upsampler.load_state_dict(load_file(hf_hub_download(LTX_REPO, UPSAMPLER_FILE)), strict=True)
    return upsampler.to(DTYPE).eval()


def inpaint_video(
        video, masks, model, prompt=None, seed=0, long_side=768, fps=24, dilate_px=15, feather_px=6,
        window=None, overlap=OVERLAP_FRAMES,
        ):
    """Regenerate the masked region; returns a video shaped like the input.

    `prompt` should describe only the masked region (None works for removal).
    `long_side` caps the stage-1 generation size; stage 2 refines at twice that.
    `dilate_px` grows the mask at stage-1 size so the fill covers the object's edges.
    `window`/`overlap` (8k + 1 frames) split long videos; window None fits the 4090 at this size.
    """
    check_video(video)
    if masks.shape != video.shape[:3]:
        raise ValueError(f"masks must be (T, H, W) = {video.shape[:3]}, got {masks.shape}")
    long_side = _stage_1_long_side(video.shape[1:3], long_side)
    window = window or _window_frames(*_generation_size(*video.shape[1:3], long_side), overlap)
    spans = _windows(len(video), window, overlap)
    encoded_prompt = _encode_prompt(model["pipe"], prompt or "")
    out = np.empty_like(video)
    context = None
    for i, (start, stop) in enumerate(spans):
        frames, gen_masks = _prepare(video[start:stop], masks[start:stop], long_side, dilate_px)
        stage_1, stage_2 = _generate(frames, gen_masks, model, encoded_prompt, seed + i, fps, context)
        real = stop - start  # drop the 8k + 1 padding frames
        stage_1, stage_2 = stage_1[:real], stage_2[:real]
        if context is not None:
            # Crossfade from the previous window's frames to this one's across the overlap.
            weight = (np.arange(1, overlap + 1, dtype=np.float32) / (overlap + 1))[:, None, None, None]
            stage_2[:overlap] = (context[1] * (1 - weight) + stage_2[:overlap] * weight).round().astype(np.uint8)
        # Overlap frames are finalized only once the next window has blended into them.
        done = real if i == len(spans) - 1 else real - overlap
        span = slice(start, start + done)
        final_masks = _resize_masks(gen_masks[:done], video.shape[1:3])
        out[span] = _composite(video[span], final_masks, stage_2[:done, ..., ::-1], feather_px)
        context = (stage_1[real - overlap:].copy(), stage_2[real - overlap:].copy())
        torch.cuda.empty_cache()
    return out


def _window_frames(height, width, overlap=OVERLAP_FRAMES):
    """Longest 8k + 1 window whose stage-2 tokens (at twice this stage-1 size) fit the VRAM budget."""
    tokens_per_latent_frame = (height * 2 // SIZE_STEP) * (width * 2 // SIZE_STEP)
    for frames in range(MAX_WINDOW_FRAMES, overlap + FRAME_STEP - 1, -FRAME_STEP):
        if tokens_per_latent_frame * ((frames - 1) // FRAME_STEP + 1) <= STAGE_2_TOKEN_BUDGET:
            return frames
    return overlap + FRAME_STEP  # over budget even at the minimum; may OOM, as unwindowed runs did


def _windows(num_frames, window, overlap):
    """(start, stop) spans covering every frame, consecutive spans sharing exactly `overlap` frames."""
    if window % FRAME_STEP != 1 or overlap % FRAME_STEP != 1:
        raise ValueError(f"window ({window}) and overlap ({overlap}) must be 8k + 1 frames")
    if window < overlap + FRAME_STEP:
        raise ValueError(f"window ({window}) must be at least overlap + {FRAME_STEP} ({overlap + FRAME_STEP})")
    spans = [(0, min(window, num_frames))]
    while spans[-1][1] < num_frames:
        start = spans[-1][1] - overlap
        spans.append((start, min(start + window, num_frames)))
    return spans


def _stage_1_long_side(size, long_side):
    """Half the source's long side, so stage 2 lands near source resolution; floored for quality."""
    return max(min(long_side, max(size) // 2), MIN_STAGE_1_LONG_SIDE)


def _generation_size(height, width, long_side):
    """Scale so the longer side is ~`long_side`, snapped to the model's size step."""
    scale = long_side / max(height, width)
    return tuple(max(SIZE_STEP, round(side * scale / SIZE_STEP) * SIZE_STEP) for side in (height, width))


def _prepare(video, masks, long_side, dilate_px=0):
    """Resize to generation size (RGB), pad the frame count to 8k + 1, and dilate the masks."""
    num_frames = len(video)
    padded = -(-(num_frames - 1) // FRAME_STEP) * FRAME_STEP + 1
    index = np.minimum(np.arange(padded), num_frames - 1)
    height, width = _generation_size(*video.shape[1:3], long_side)
    frames = np.stack([
        cv2.cvtColor(cv2.resize(video[i], (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for i in index
    ])
    # Area-resize then threshold at zero, so thin masked structures survive the downscale.
    binary = (np.asarray(masks) > 0).astype(np.float32)
    gen_masks = np.stack([
        _dilate(cv2.resize(binary[i], (width, height), interpolation=cv2.INTER_AREA) > 0, dilate_px)
        for i in index
    ])
    return frames, gen_masks


def _dilate(mask, px):
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _resize_masks(masks, size):
    height, width = size
    return np.stack([
        cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.5 for m in masks
    ])


@torch.inference_mode()
def _encode_prompt(pipe, prompt):
    """(prompt embeds, mask, negative embeds, mask); the negative is used only by dev's guided stage 1."""
    return pipe.encode_prompt(prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, device="cuda")


@torch.inference_mode()
def _generate(frames, gen_masks, model, encoded_prompt, seed, fps, context=None):
    """Two-stage LTX inpaint; returns RGB (stage-1 composite (T, H, W, 3), stage 2 (T, 2H, 2W, 3)).

    `context` = (stage-1, stage-2) frames from the previous window; they are pinned as this window's start.
    """
    pipe, distilled = model["pipe"], model["distilled"]
    num_frames, height, width = gen_masks.shape
    generator = torch.Generator("cuda").manual_seed(seed)
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = encoded_prompt
    pins = [None, None]
    if context is not None:
        # The overlap shows the previous result un-greened, and conditioning keeps its latents clean.
        frames, gen_masks = frames.copy(), gen_masks.copy()
        frames[: len(context[0])], gen_masks[: len(context[0])] = context[0], False
        pins = [[LTX2VideoCondition(frames=f, index=0, strength=1.0)] for f in context]
    shared = dict(
        prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_mask, num_frames=num_frames, frame_rate=fps,
        generator=generator, output_type="latent", return_dict=False,
    )

    # Stage 1: the reference shows the scene with the region to fill painted green.
    reference = frames.copy()
    reference[gen_masks] = GREEN_RGB
    if distilled:
        schedule = dict(sigmas=DISTILLED_SIGMA_VALUES, num_inference_steps=len(DISTILLED_SIGMA_VALUES), **NO_GUIDANCE)
    else:
        schedule = dict(
            num_inference_steps=DEV_STEPS, negative_prompt_embeds=negative_embeds,
            negative_prompt_attention_mask=negative_mask, **DEV_GUIDANCE,
        )
    video_latent, audio_latent = pipe(
        reference_conditions=[LTX2ReferenceCondition(frames=[Image.fromarray(f) for f in reference])],
        conditions=pins[0], reference_downscale_factor=1, width=width, height=height, **schedule, **shared,
    )

    # Stage 2: keep the original outside the mask, upsample 2x in latent space, then refine.
    stage_1 = np.where(gen_masks[..., None], _decode(pipe, video_latent), frames)
    upsampled = model["upsample_pipe"](
        video=[Image.fromarray(f) for f in stage_1], height=height, width=width,
        generator=generator, output_type="latent", return_dict=False,
    )[0]
    with _refining(model):
        refined, _ = pipe(
            latents=upsampled, audio_latents=audio_latent, conditions=pins[1], width=width * 2, height=height * 2,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, num_inference_steps=len(STAGE_2_DISTILLED_SIGMA_VALUES),
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], **NO_GUIDANCE, **shared,
        )
    return stage_1, _decode(pipe, refined)


@contextlib.contextmanager
def _refining(model):
    """Stage 2 setup: the unshifted scheduler, and the IC-LoRA swapped for the distilled LoRA (none on distilled)."""
    pipe, distilled = model["pipe"], model["distilled"]
    scheduler = pipe.scheduler
    try:
        pipe.scheduler = model["refine_scheduler"]
        if distilled:
            pipe.disable_lora()
        else:
            pipe.set_adapters(["distilled"], [1.0])
        yield
    finally:
        pipe.scheduler = scheduler
        if distilled:
            pipe.enable_lora()
        else:
            pipe.set_adapters(["inpaint"], [1.0])


def _decode(pipe, latents):
    """Pipeline latents (B, C, F, H, W) -> (F, H, W, 3) uint8 RGB."""
    video = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    return (np.clip(video, 0, 1) * 255).round().astype(np.uint8)


def _composite(video, masks, generated, feather_px):
    """Blend generated pixels into the original through an outward-feathered mask, at the original size."""
    height, width = video.shape[1:3]
    out = np.empty_like(video)
    for i, (original, mask) in enumerate(zip(video, masks)):
        gen = cv2.resize(generated[i], (width, height), interpolation=cv2.INTER_LANCZOS4)
        alpha = (np.asarray(mask) > 0).astype(np.float32)
        if feather_px > 0:
            # Grow then blur, so the ramp starts at ~1 on the mask edge instead of stepping down.
            alpha = np.maximum(alpha, cv2.GaussianBlur(_dilate(alpha, feather_px).astype(np.float32), (0, 0), feather_px))
        alpha = alpha[..., None]
        out[i] = (gen * alpha + original * (1 - alpha)).round().astype(np.uint8)
    return out
