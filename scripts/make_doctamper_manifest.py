import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


MASK_SUFFIXES = ["", "_mask", "-mask", "_gt", "-gt", "_label", "-label", "_seg", "-seg"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def find_mask(masks_dir: Optional[Path], img_path: Path) -> Optional[Path]:
    if masks_dir is None:
        return None
    stem = img_path.stem
    # Try exact filename first.
    exact = masks_dir / img_path.name
    if exact.exists():
        return exact
    # Try suffix variations with common extensions.
    for suf in MASK_SUFFIXES:
        for ext in MASK_EXTS:
            candidate = masks_dir / f"{stem}{suf}{ext}"
            if candidate.exists():
                return candidate
    return None


def collect_split(split_dir: Path) -> List[Tuple[Path, Optional[Path], int]]:
    """
    Collect (image, mask, label) tuples for a DocTamper split.
    Assumes subfolders named 'images' and 'masks' (case-insensitive).
    """
    images_dir = next((p for p in split_dir.iterdir() if p.is_dir() and p.name.lower() == "images"), None)
    masks_dir = next((p for p in split_dir.iterdir() if p.is_dir() and p.name.lower() == "masks"), None)
    if images_dir is None:
        raise FileNotFoundError(f"No images/ folder found in {split_dir}")

    items: List[Tuple[Path, Optional[Path], int]] = []
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        mask_path = find_mask(masks_dir, img_path)
        label = 0 if mask_path is not None else 1
        items.append((img_path, mask_path, label))
    return items


def write_manifest(entries: List[Tuple[Path, Optional[Path], int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img, mask, label in entries:
            record = {"image": str(img.resolve()), "label": int(label)}
            if mask is not None:
                record["mask"] = str(mask.resolve())
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifests for DocTamper dataset (images/masks folders).")
    parser.add_argument("--root", type=Path, required=True, help="Path to doctamper-dataset root")
    parser.add_argument(
        "--train-splits",
        type=str,
        default="DocTamper Training,DocTamper SCD,DocTamper FCD",
        help="Comma-separated split folder names to use for training",
    )
    parser.add_argument(
        "--val-splits",
        type=str,
        default="DocTamper Testing",
        help="Comma-separated split folder names to use for validation",
    )
    parser.add_argument("--train-out", type=Path, default=Path("data/train_manifest.jsonl"))
    parser.add_argument("--val-out", type=Path, default=Path("data/val_manifest.jsonl"))
    args = parser.parse_args()

    train_splits = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    val_splits = [s.strip() for s in args.val_splits.split(",") if s.strip()]

    def gather(split_names: List[str]) -> List[Tuple[Path, Optional[Path], int]]:
        collected: List[Tuple[Path, Optional[Path], int]] = []
        for name in split_names:
            split_dir = args.root / name
            if not split_dir.exists():
                raise FileNotFoundError(f"Split folder not found: {split_dir}")
            collected.extend(collect_split(split_dir))
        return collected

    train_entries = gather(train_splits)
    val_entries = gather(val_splits)

    write_manifest(train_entries, args.train_out)
    write_manifest(val_entries, args.val_out)

    train_masks = sum(1 for _, m, _ in train_entries if m is not None)
    val_masks = sum(1 for _, m, _ in val_entries if m is not None)
    print(f"Wrote {len(train_entries)} train ({train_masks} with masks) and {len(val_entries)} val ({val_masks} with masks) entries.")


if __name__ == "__main__":
    main()
