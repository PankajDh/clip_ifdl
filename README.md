# CLIP-IFDL (Noise-assisted Prompt Learning for Document Tampering)

This repository implements the ECCV 2024 paper “Noise-assisted Prompt Learning for Image Forgery Detection and Localization” (CLIP-IFDL) and adapts it for business document tampering detection/localization.

## What's in the repo
- `clip_ifdl/models/clip_ifdl.py`: Main CLIP-IFDL model with Instance-aware Dual-stream Prompt Learning (IDPL) and Forgery-enhanced Noise Adapter (FENA).
- `clip_ifdl/models/idpl.py`: Prompt learning modules (PANet + EANet) and similarity head.
- `clip_ifdl/models/fena.py`: BayarConv-based noise extractor and zero-initialized adapters injected into CLIP ViT layers.
- `clip_ifdl/data/dataset.py`: JSONL-manifest dataset for document images/masks.
- `clip_ifdl/data/transforms.py`: Deterministic resize + normalization aligned with CLIP.
- `clip_ifdl/config/default.yaml`: Example training config.
- `train.py`: Reference training loop.

## Environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If you prefer a specific CLIP variant, adjust `model.clip_model` and `model.clip_pretrained` in the config (OpenAI ViT-B/16 by default).

## Data
The training/validation manifests are JSONL files with one entry per image:
```json
{"image": "/path/doc.png", "mask": "/path/doc_mask.png", "label": 0}
```
- `label`: `1` for authentic, `0` for tampered.
- `mask`: optional; when absent, localization loss is skipped for that sample.

## Training
```
python3 train.py --config clip_ifdl/config/default.yaml      # standard
python3 train.py --config clip_ifdl/config/low_mem.yaml      # smaller VRAM
```
If you are using the Kaggle dataset `dinmkeljiame/doctamper`, first download/unzip it, then build manifests:
```
python3 scripts/make_manifest.py \
  --data-root /path/to/doctamper \
  --mask-suffix "_mask" \
  --mask-dirs "masks" \
  --val-ratio 0.1 \
  --train-out data/train_manifest.jsonl \
  --val-out data/val_manifest.jsonl
```
Adjust `--mask-suffix`/`--mask-dirs` to match the dataset’s mask naming/layout. The config already points to `data/train_manifest.jsonl` and `data/val_manifest.jsonl`.

Key config knobs:
- `model.adapter_layers`: transformer layers where FENA is applied (supports 0..12 for ViT-B/16).
- `model.prompt_tokens`: number of learnable prompt tokens per stream (real/fake).
- `optimization.lr` and `grad_clip`: stability controls when unfreezing only adapters/prompts.
- `optimization.use_amp`: mixed precision on/off; keep it on for 24 GB-class GPUs (e.g., L4).

### Memory notes
- For 24 GB (e.g., L4/3090/4090): default config (512px, batch 4) with AMP should fit.
- For ~16 GB: use `clip_ifdl/config/low_mem.yaml` (448px, batch 2) or drop to 384px if needed.

## Notes on implementation choices
- CLIP weights remain frozen to preserve priors; only prompts/adapters/decoder train.
- FENA fuses BayarConv residuals with frequency magnitude cues, patchifies them to CLIP’s token grid, and injects via zero-initialized cross-attention gates.
- IDPL seeds prompts from text embeddings (“real/forged document”), adapts them per-instance using CLS tokens (PANet) and visual-token cross-attention (EANet), then scores real vs. fake via cosine similarity.
- Localization head is a lightweight projection from patch tokens to a full-resolution mask; swap with a richer decoder if you need sharper boundaries.

## Next steps
- Replace the simple decoder with a multi-scale transformer/conv decoder for finer boundaries on high-res docs.
- Add data augmentations tailored for documents (blur/compression/resizing/print-scan artifacts).
- Incorporate class-balancing samplers if real images dominate the corpus.
