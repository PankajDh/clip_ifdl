import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_ifdl.data.dataset import DocForgeryDataset
from clip_ifdl.data.transforms import default_transform
from clip_ifdl.models.clip_ifdl import CLIPIFDL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch: List[Dict[str, Any]]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    masks_raw = [b["mask"] for b in batch]
    has_mask = any(m is not None for m in masks_raw)
    masks: Optional[torch.Tensor] = None
    if has_mask:
        template = masks_raw[0] if masks_raw[0] is not None else torch.zeros_like(batch[0]["image"][:1])
        stacked: List[torch.Tensor] = []
        for m in masks_raw:
            if m is None:
                stacked.append(torch.zeros_like(template))
            else:
                stacked.append(m)
        masks = torch.stack(stacked, dim=0)
    return images, labels, masks


def build_dataloaders(cfg: Dict[str, Any]):
    size = cfg["data"]["image_size"]

    def _transform(image, mask):
        return default_transform(image, mask, image_size=size)

    train_ds = DocForgeryDataset(cfg["data"]["train_manifest"], transform=_transform, image_size=size)
    val_ds = DocForgeryDataset(cfg["data"]["val_manifest"], transform=_transform, image_size=size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["optimization"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["optimization"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str, step: int) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = Path(ckpt_dir) / f"model_{step}.pt"
    torch.save(state, path)


def train(cfg: Dict[str, Any]) -> None:
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)

    model = CLIPIFDL(
        model_name=cfg["model"]["clip_model"],
        pretrained=cfg["model"]["clip_pretrained"],
        prompt_tokens=cfg["model"]["prompt_tokens"],
        adapter_heads=cfg["model"]["adapter_heads"],
        adapter_layers=cfg["model"]["adapter_layers"],
        noise_channels=cfg["model"]["noise_channels"],
        dropout=cfg["model"]["dropout"],
        device=str(device),
    ).to(device)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_loc = nn.BCEWithLogitsLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg["optimization"]["lr"], weight_decay=cfg["optimization"]["weight_decay"])

    scaler = amp.GradScaler("cuda", enabled=cfg["optimization"].get("use_amp", False))
    use_amp = scaler.is_enabled()
    global_step = 0
    for epoch in range(cfg["optimization"]["max_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, labels, masks in pbar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            if masks is not None:
                masks = masks.to(device, dtype=torch.float32)

            with amp.autocast("cuda", enabled=use_amp):
                outputs = model(images, image_size=cfg["data"]["image_size"])
                cls_loss = criterion_cls(outputs["det_logit"].view(-1), labels)

                loc_loss = torch.tensor(0.0, device=device)
                if masks is not None:
                    loc_loss = criterion_loc(outputs["loc_logits"], masks)

                loss = cls_loss + loc_loss

            scaler.scale(loss).backward()
            if cfg["optimization"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, cfg["optimization"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            if global_step % cfg["logging"]["log_every"] == 0:
                pbar.set_postfix({"loss": loss.item(), "cls": cls_loss.item(), "loc": loc_loss.item()})

        # Simple validation
        model.eval()
        val_cls, val_loc, count = 0.0, 0.0, 0
        with torch.no_grad(), amp.autocast("cuda", enabled=use_amp):
            for images, labels, masks in val_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                if masks is not None:
                    masks = masks.to(device, dtype=torch.float32)
                outputs = model(images, image_size=cfg["data"]["image_size"])
                cls_loss = criterion_cls(outputs["det_logit"].view(-1), labels)
                loc_loss = torch.tensor(0.0, device=device)
                if masks is not None:
                    loc_loss = criterion_loc(outputs["loc_logits"], masks)
                val_cls += cls_loss.item() * images.size(0)
                val_loc += loc_loss.item() * images.size(0)
                count += images.size(0)
        avg_cls = val_cls / max(count, 1)
        avg_loc = val_loc / max(count, 1)
        print(f"[val] epoch={epoch} cls={avg_cls:.4f} loc={avg_loc:.4f}")

        save_checkpoint({"model": model.state_dict(), "cfg": cfg}, cfg["logging"]["ckpt_dir"], epoch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP-IFDL for document forgery detection")
    parser.add_argument("--config", type=str, default="clip_ifdl/config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    train(cfg)


if __name__ == "__main__":
    main()
