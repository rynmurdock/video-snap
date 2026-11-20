"""Mask-conditioned video inpainting with the LTX-2.3 In/Outpainting IC-LoRA.

Videos are cv2 arrays: (T, H, W, 3) uint8 BGR. Masks are (T, H, W), nonzero = inpaint.
Follows diffusers' LTX-2 two-stage recipe: IC-LoRA generation from a green-keyed reference, 2x latent upsample,
then a 3-step refine (dev + distilled LoRA by default; the distilled model with `distilled=True`).
"""
import contextlib

import cv2
import numpy as np
import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, LTX2InContextPipeline, LTX2VideoTransformer3DModel
)
from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES
from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
from diffusers.models.latent_upscaler.latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition, retrieve_latents
from diffusers.pipelines.ltx2.utils import (
    DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
)
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Gemma3ForConditionalGeneration

from lib.video_io import check_video

# Quantized weights: the full-precision model (95 GB, 38 GB transformer) does not fit a 4090.
BASE_REPO = "diffusers/LTX-2.3-Diffusers"
DISTILLED_BASE_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
TRANSFORMER_REPO = "QuantStack/LTX-2.3-GGUF"
TRANSFORMER_FILE = "LTX-2.3-dev/LTX-2.3-dev-Q4_K_M.gguf"  # no 1.1 dev release exists; distilled is 1.1
DISTILLED_TRANSFORMER_FILE = "LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
TEXT_ENCODER_REPO = "unsloth/gemma-3-12b-it-qat-bnb-4bit"
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


def get_video_model(model_path="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting", distilled=False):
    """Load the inpainting models; `model_path` is the IC-LoRA repo, `distilled` picks the fast base model."""
    base_repo = DISTILLED_BASE_REPO if distilled else BASE_REPO
    transformer = _load_transformer(DISTILLED_TRANSFORMER_FILE if distilled else TRANSFORMER_FILE, base_repo)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(TEXT_ENCODER_REPO, dtype=DTYPE)
    pipe = LTX2InContextPipeline.from_pretrained(
        base_repo, transformer=transformer, text_encoder=text_encoder, dtype=DTYPE
    )
    # Kept as togglable adapters (not fused): stage 1 uses only the IC-LoRA, stage 2 only the distilled one.
    _load_lora(pipe, model_path, LORA_FILE, "inpaint")
    if not distilled:
        _load_lora(pipe, LTX_REPO, DISTILLED_LORA_FILE, "distilled")
        pipe.set_adapters(["inpaint"], [1.0])
    pipe.vae.enable_tiling()
    _offload(pipe)
    # As in diffusers' recipe: stage 2 sigmas are used as given, without the dev scheduler's time shift.
    refine_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
    )
    return {"pipe": pipe, "distilled": distilled, "refine_scheduler": refine_scheduler, "upsampler": _load_upsampler()}


def _load_lora(pipe, repo, weight_name, adapter_name):
    """Load a LoRA and check every layer in the file attached; unmatched keys only log a warning."""
    pipe.load_lora_weights(repo, weight_name=weight_name, adapter_name=adapter_name)
    with safe_open(hf_hub_download(repo, weight_name), "pt") as f:
        expected = sum(key.endswith("lora_A.weight") for key in f.keys())
    attached = sum(adapter_name in getattr(m, "lora_A", {}) for m in pipe.transformer.modules())
    if attached != expected:
        raise RuntimeError(f"{weight_name} attached to {attached} of its {expected} layers")


def _offload(pipe):
    """Stream the big models (dev + distilled LoRA alone exceed 24 GB); keep the small ones on the GPU."""
    for module in (pipe.transformer, pipe.connectors):
        module.enable_group_offload(onload_device=torch.device("cuda"), offload_type="leaf_level", use_stream=True)
    for module in (pipe.vae, pipe.audio_vae, pipe.vocoder):
        module.to("cuda")
    # The 4-bit text encoder is placed on the GPU by bitsandbytes at load and cannot be moved.


def _load_upsampler():
    """LTX-2.3's 2x latent upsampler; its single file uses a plain conv resampler, not diffusers' rational one."""
    upsampler = LTX2LatentUpsamplerModel(
        in_channels=128, mid_channels=1024, num_blocks_per_stage=4, dims=3,
        spatial_upsample=True, temporal_upsample=False, use_rational_resampler=False,
    )
    upsampler.load_state_dict(load_file(hf_hub_download(LTX_REPO, UPSAMPLER_FILE)), strict=True)
    return upsampler.to(DTYPE).eval()


def _convert_ltx23_transformer(checkpoint, **kwargs):
    """diffusers' single-file converter predates LTX-2.3: it misses the prompt_adaln rename."""
    converted = convert_ltx2_transformer_to_diffusers(checkpoint, **kwargs)
    return {key.replace("prompt_adaln_single", "prompt_adaln"): value for key, value in converted.items()}


def _load_transformer(transformer_file, base_repo):
    # Patches diffusers' global converter entry; the rename is a no-op for older LTX-2 checkpoints.
    loadable = SINGLE_FILE_LOADABLE_CLASSES["LTX2VideoTransformer3DModel"]
    loadable["checkpoint_mapping_fn"] = _convert_ltx23_transformer
    return LTX2VideoTransformer3DModel.from_single_file(
        hf_hub_download(TRANSFORMER_REPO, transformer_file),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        config=base_repo,
        subfolder="transformer",
        dtype=DTYPE,
    )


def inpaint_video(video, masks, model, prompt=None, seed=0, long_side=512, fps=24, dilate_px=15, feather_px=6):
    """Regenerate the masked region; returns a video shaped like the input.

    `prompt` should describe only the masked region (None works for removal).
    `long_side` caps the stage-1 generation size; stage 2 refines at twice that.
    `dilate_px` grows the mask at stage-1 size so the fill covers the object's edges.
    """
    check_video(video)
    if masks.shape != video.shape[:3]:
        raise ValueError(f"masks must be (T, H, W) = {video.shape[:3]}, got {masks.shape}")
    long_side = _stage_1_long_side(video.shape[1:3], long_side)
    frames, gen_masks = _prepare(video, masks, long_side, dilate_px)
    generated = _generate(frames, gen_masks, model, prompt or "The scene", seed, fps)
    return _composite(video, _resize_masks(gen_masks[: len(video)], video.shape[1:3]), generated, feather_px)


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
def _generate(frames, gen_masks, model, prompt, seed, fps):
    """Two-stage LTX inpaint; returns (T, 2H, 2W, 3) uint8 BGR at twice the generation size."""
    pipe, distilled = model["pipe"], model["distilled"]
    num_frames, height, width = gen_masks.shape
    generator = torch.Generator("cuda").manual_seed(seed)
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = pipe.encode_prompt(
        prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, do_classifier_free_guidance=not distilled, device="cuda"
    )
    shared = dict(
        prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_mask, num_frames=num_frames, frame_rate=fps,
        generator=generator, output_type="latent", return_dict=False,
    )

    # Stage 1: the reference shows the scene with the region to fill painted green.
    reference = frames.copy()
    reference[gen_masks] = GREEN_RGB
    if distilled:
        stage_1 = dict(sigmas=DISTILLED_SIGMA_VALUES, num_inference_steps=len(DISTILLED_SIGMA_VALUES), **NO_GUIDANCE)
    else:
        stage_1 = dict(
            num_inference_steps=DEV_STEPS, negative_prompt_embeds=negative_embeds,
            negative_prompt_attention_mask=negative_mask, **DEV_GUIDANCE,
        )
    video_latent, audio_latent = pipe(
        reference_conditions=[LTX2ReferenceCondition(frames=[Image.fromarray(f) for f in reference])],
        reference_downscale_factor=1, width=width, height=height, **stage_1, **shared,
    )

    # Stage 2: keep the original outside the mask, upsample 2x in latent space, then refine.
    composited = np.where(gen_masks[..., None], _decode(pipe, video_latent), frames)
    upsampler = model["upsampler"].to("cuda")
    try:
        upsampled = upsampler(_encode(pipe, composited).to(DTYPE))
    finally:
        upsampler.to("cpu")
    with _refining(model):
        refined, _ = pipe(
            latents=upsampled, audio_latents=audio_latent, width=width * 2, height=height * 2,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, num_inference_steps=len(STAGE_2_DISTILLED_SIGMA_VALUES),
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], **NO_GUIDANCE, **shared,
        )
    return np.ascontiguousarray(_decode(pipe, refined)[..., ::-1])


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


def _encode(pipe, frames):
    """(F, H, W, 3) uint8 RGB frames -> latents in the form the pipeline returns and accepts."""
    pixels = pipe.video_processor.preprocess_video([Image.fromarray(f) for f in frames])
    return retrieve_latents(pipe.vae.encode(pixels.to("cuda", pipe.vae.dtype)), sample_mode="argmax")


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
