
# Setup

```
uv venv
source ./.venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
hf auth login
```

You'll have to accept the IC-LoRA & SAM3 licenses:
[IC-LoRA license](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting)
[SAM3 license](https://huggingface.co/facebook/sam3)

# Inpaint a video

```
python -m run.process_one_video --video input.mp4 --masklet-prompt "<sam3 prompt>" --save-masks body.mp4 --out out.mp4 --prompt "<ltx prompt specifying only the masked content>"
```

- `--masks` is a mask video: white specifies the region to regenerate.
- Whole videos by default (`--max-frames N` to trim); long ones run in overlapping windows sized to fit a 4090.
- Output is silent mp4v at the source fps; for variable-frame-rate input, convert first: `ffmpeg -i in.mp4 -vf fps=24 cfr.mp4`.

# Smoke test

```
python -m tests.smoke          # imports, shapes, mask files: seconds, no GPU
python -m tests.smoke --full   # runs SAM3 + LTX (GPU, gated weights): minutes
```
