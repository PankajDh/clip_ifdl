import argparse
from pathlib import Path

import cv2
import torch

from clip_ifdl.data.transforms import default_transform
from clip_ifdl.models.clip_ifdl import CLIPIFDL


def load_model(ckpt_path: str, device: torch.device, image_size: int):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {}) if cfg else {}
    model = CLIPIFDL(
        model_name=model_cfg.get("clip_model", "ViT-B-16"),
        pretrained=model_cfg.get("clip_pretrained", "openai"),
        prompt_tokens=model_cfg.get("prompt_tokens", 8),
        adapter_heads=model_cfg.get("adapter_heads", 8),
        adapter_layers=model_cfg.get("adapter_layers", [0, 3, 6, 9, 12]),
        noise_channels=model_cfg.get("noise_channels", 64),
        dropout=model_cfg.get("dropout", 0.0),
        device=str(device),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--out-mask", type=str, help="Optional path to save predicted mask PNG")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.image_size)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor, _ = default_transform(img, None, image_size=args.image_size)
    tensor = tensor.unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        out = model(tensor, image_size=args.image_size)
        prob_real = out["det_logit"].sigmoid().item()
        mask = out["loc_logits"].sigmoid().cpu().numpy()[0, 0]  # HxW

    print(f"Real probability: {prob_real:.4f} (forged prob: {1 - prob_real:.4f})")

    if args.out_mask:
        mask_img = (mask * 255).astype("uint8")
        cv2.imwrite(args.out_mask, mask_img)
        print(f"Saved mask to {args.out_mask}")


if __name__ == "__main__":
    main()
