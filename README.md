
# Setup

uv venv
source ./.venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# Inpaint a video

python -m run.process_one_video --video in.mp4 --masks masks.mp4 --out out.mp4

- `--masks` is a mask video: white = region to regenerate.
- `--prompt` is optional & should describe only the masked region.
- First run downloads ~37 GB of weights to the Hugging Face cache; accept the gated
  [IC-LoRA license](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting) & `hf auth login` first.

# Masklets from text prompts (SAM3)

python -m run.process_one_video --video in.mp4 --masklet-prompt "dog" --prompt "a red fox" --out out.mp4
python -m run.make_masklets --video in.mp4 --prompt "dog" --out dog.mp4

- Accept the gated [SAM3 license](https://huggingface.co/facebook/sam3) first.
- Repeat `--masklet-prompt` (or `--prompt` in `make_masklets`) & pass several `--masks` to compose them.
- `--compose union|intersect|subtract`; subtract = first minus the rest, with `--masks` ordered before prompts:

python -m run.process_one_video --video in.mp4 --masks person.mp4 --masklet-prompt "face" --compose subtract --save-masks body.mp4 --out out.mp4

# Smoke test

python -m tests.smoke          # imports + shapes, seconds, no GPU
python -m tests.smoke --full   # loads the real model, minutes
