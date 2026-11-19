
# Setup

uv venv
source ./.venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
hf auth login   # after accepting the IC-LoRA & SAM3 licenses linked below

# Inpaint a video

python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out out.mp4

- `--masks` is a mask video: white = region to regenerate.
- `--prompt` is optional & should describe only the masked region.
- First run downloads ~37 GB of weights to the Hugging Face cache; accept the gated
  [IC-LoRA license](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting) first.
- Output is silent mp4v at the source fps; for variable-frame-rate input, convert first: `ffmpeg -i in.mp4 -vf fps=24 cfr.mp4`.

# Masklets from text prompts (SAM3)

python -m run.process_one_video --video in.mp4 --masklet-prompt "dog" --prompt "a red fox" --out out.mp4
python -m run.make_masklets --video in.mp4 --masklet-prompt "dog" --out dog.mp4

- Accept the gated [SAM3 license](https://huggingface.co/facebook/sam3) first; SAM3 fetches its `kernels` once online.
- Repeat `--masklet-prompt` & pass several `--masks` to compose them.
- `--compose union|intersect|subtract`; subtract = first minus the rest, with `--masks` ordered before prompts:

python -m run.process_one_video --video /home/ryn_mote/Misc/INTERMARCHE_TEST.mp4 --masklet-prompt "person" --save-masks body.mp4 --out out.mp4

# Smoke test

python -m tests.smoke          # imports, shapes, mask files: seconds, no GPU
python -m tests.smoke --full   # runs SAM3 + LTX (GPU, gated weights): minutes
