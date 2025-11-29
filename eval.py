import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_ifdl.data.dataset import DocForgeryDataset
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
    parser = argparse.ArgumentParser(description="Run inference on a manifest and save predictions.")
    parser.add_argument("--manifest", type=str, required=True, help="JSONL manifest for inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save predicted masks and scores")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-masks", action="store_true", help="Save predicted mask PNGs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.image_size)

    def _transform(image, mask):
        return default_transform(image, mask, image_size=args.image_size)

    ds = DocForgeryDataset(args.manifest, transform=_transform, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.txt"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        with scores_path.open("w", encoding="utf-8") as sf:
            for batch in tqdm(loader, desc="infer"):
                images = batch["image"].to(device, dtype=torch.float32)
                outputs = model(images, image_size=args.image_size)
                det_prob = outputs["det_logit"].sigmoid()  # probability of "real"
                for i, prob in enumerate(det_prob.tolist()):
                    idx = len(list(out_dir.glob("*.png"))) + i
                    sf.write(f"{idx}\t{prob}\n")
                if args.save_masks:
                    import cv2
                    import numpy as np

                    masks = outputs["loc_logits"].sigmoid().cpu().numpy()  # Bx1xHxW
                    for i, mask in enumerate(masks):
                        mask_img = (mask[0] * 255).astype(np.uint8)
                        out_path = out_dir / f"mask_{len(list(out_dir.glob('mask_*.png')))+i:06d}.png"
                        cv2.imwrite(str(out_path), mask_img)


if __name__ == "__main__":
    main()
