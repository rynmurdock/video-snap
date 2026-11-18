"""Mask-conditioned video inpainting with the LTX-2.3 In/Outpainting IC-LoRA.

Videos are cv2 arrays: (T, H, W, 3) uint8 BGR. Masks are (T, H, W), nonzero = inpaint.
"""
import cv2
import numpy as np
import torch
from diffusers import (
    GGUFQuantizationConfig,
    LTX2InContextPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2LatentUpsamplerModel,
    LTX2VideoTransformer3DModel,
)
from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES
from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import Gemma3ForConditionalGeneration

# Quantized weights: the full-precision model (95 GB, 38 GB transformer) does not fit a 4090.
BASE_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
TRANSFORMER_REPO = "QuantStack/LTX-2.3-GGUF"
TRANSFORMER_FILE = "LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
TEXT_ENCODER_REPO = "unsloth/gemma-3-12b-it-qat-bnb-4bit"
UPSAMPLER_REPO = "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
LORA_FILE = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

DTYPE = torch.bfloat16
FRAME_STEP = 8  # the VAE compresses time 8x, so frame counts must be 8k + 1
SIZE_STEP = 32


def get_video_model(model_path="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting"):
    """Load the inpainting pipeline; `model_path` is the IC-LoRA repo."""
    transformer = _load_transformer()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(TEXT_ENCODER_REPO, dtype=DTYPE)
    pipe = LTX2InContextPipeline.from_pretrained(
        BASE_REPO, transformer=transformer, text_encoder=text_encoder, dtype=DTYPE
    )
    # Kept as a togglable adapter (not fused): stage 2 runs on the bare distilled model.
    pipe.load_lora_weights(model_path, weight_name=LORA_FILE, adapter_name="inpaint")
    # A LoRA whose keys match nothing only logs a warning, so check that it attached.
    if "inpaint" not in pipe.get_list_adapters().get("transformer", []):
        raise RuntimeError(f"IC-LoRA from {model_path} did not attach to the transformer")
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        UPSAMPLER_REPO, subfolder="latent_upsampler", dtype=DTYPE
    )
    # Small enough to stay on the GPU; offloading it too would steal the shared VAE's offload hook.
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler.to("cuda"))
    return {"pipe": pipe, "upsample_pipe": upsample_pipe}


def _convert_ltx23_transformer(checkpoint, **kwargs):
    """diffusers' single-file converter predates LTX-2.3: it misses the prompt_adaln rename."""
    converted = convert_ltx2_transformer_to_diffusers(checkpoint, **kwargs)
    return {key.replace("prompt_adaln_single", "prompt_adaln"): value for key, value in converted.items()}


def _load_transformer():
    loadable = SINGLE_FILE_LOADABLE_CLASSES["LTX2VideoTransformer3DModel"]
    loadable["checkpoint_mapping_fn"] = _convert_ltx23_transformer
    return LTX2VideoTransformer3DModel.from_single_file(
        hf_hub_download(TRANSFORMER_REPO, TRANSFORMER_FILE),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        config=BASE_REPO,
        subfolder="transformer",
        dtype=DTYPE,
    )


def inpaint_video(
        video,
        masks,
        model,
        prompt=None,
        seed=0,
        long_side=768,
        fps=24,
        feather_px=6,
        ):
    """Regenerate the masked region; returns a video shaped like the input.

    `prompt` should describe only the masked region (None works for removal).
    `long_side` is the stage-1 generation size; stage 2 refines at twice that.
    """
    _check_inputs(video, masks)
    frames, gen_masks = _prepare(video, masks, long_side)
    generated = _generate(frames, gen_masks, model, prompt or "", seed, fps)
    return _composite(video, masks, generated[: len(video)], feather_px)


def _check_inputs(video, masks):
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError(f"video must be (T, H, W, 3) uint8 BGR, got {video.shape} {video.dtype}")
    if masks.shape != video.shape[:3]:
        raise ValueError(f"masks must be (T, H, W) = {video.shape[:3]}, got {masks.shape}")


def _generation_size(height, width, long_side):
    """Scale so the longer side is ~`long_side`, snapped to the model's size step."""
    scale = long_side / max(height, width)
    return tuple(max(SIZE_STEP, round(side * scale / SIZE_STEP) * SIZE_STEP) for side in (height, width))


def _prepare(video, masks, long_side):
    """Resize to generation size (RGB) and pad the frame count to 8k + 1 by repeating the last frame."""
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
        cv2.resize(binary[i], (width, height), interpolation=cv2.INTER_AREA) > 0 for i in index
    ])
    return frames, gen_masks


@torch.inference_mode()
def _generate(frames, gen_masks, model, prompt, seed, fps):
    """Two-stage LTX inpaint; returns (T, 2H, 2W, 3) uint8 BGR at twice the generation size."""
    pipe, upsample_pipe = model["pipe"], model["upsample_pipe"]
    num_frames, height, width = gen_masks.shape
    generator = torch.Generator("cuda").manual_seed(seed)
    shared = dict(
        prompt=prompt, negative_prompt="", num_frames=num_frames, frame_rate=fps,
        guidance_scale=1.0, stg_scale=0.0, audio_guidance_scale=1.0, audio_stg_scale=0.0,
        generator=generator, return_dict=False,
    )

    # Stage 1: generated tokens may not attend to the reference inside the mask.
    attention_mask = torch.from_numpy(1.0 - gen_masks.astype(np.float32))[None, None]
    pipe.enable_lora()
    video_latent, audio_latent = pipe(
        reference_conditions=[LTX2ReferenceCondition(frames=[Image.fromarray(f) for f in frames])],
        conditioning_attention_mask=attention_mask, reference_downscale_factor=1,
        width=width, height=height, sigmas=DISTILLED_SIGMA_VALUES,
        num_inference_steps=len(DISTILLED_SIGMA_VALUES), output_type="latent", **shared,
    )

    # Stage 2: 2x latent upsample, then a short refine without the LoRA or reference.
    upsampled = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]
    pipe.disable_lora()
    try:
        refined, _ = pipe(
            latents=upsampled, audio_latents=audio_latent, width=width * 2, height=height * 2,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            num_inference_steps=len(STAGE_2_DISTILLED_SIGMA_VALUES), output_type="np", **shared,
        )
    finally:
        pipe.enable_lora()
    rgb = (np.clip(refined[0], 0, 1) * 255).round().astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def _composite(video, masks, generated, feather_px):
    """Blend generated pixels into the original through an outward-feathered mask, at the original size."""
    height, width = video.shape[1:3]
    out = np.empty_like(video)
    for i, (original, mask, gen) in enumerate(zip(video, masks, generated)):
        gen = cv2.resize(gen, (width, height), interpolation=cv2.INTER_LANCZOS4)
        alpha = (np.asarray(mask) > 0).astype(np.float32)
        if feather_px > 0:
            # Feather outwards only: inside the mask stays fully generated.
            alpha = np.maximum(alpha, cv2.GaussianBlur(alpha, (0, 0), feather_px))
        alpha = alpha[..., None]
        out[i] = (gen * alpha + original * (1 - alpha)).round().astype(np.uint8)
    return out
