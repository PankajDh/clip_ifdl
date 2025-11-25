import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_pairs(
    root: Path,
    mask_suffix: str = "_mask",
    mask_dirs: Optional[List[str]] = None,
) -> List[Tuple[Path, Optional[Path], int]]:
    """
    Heuristic pairing:
    - If a mask with the same stem + mask_suffix exists, label=0 (tampered).
    - Otherwise label=1 (authentic).
    - Optionally restrict mask search to specific subdirs.
    """
    pairs: List[Tuple[Path, Optional[Path], int]] = []
    mask_dirs_paths = [root / d for d in (mask_dirs or [])]

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        candidate_masks: List[Path] = []

        # Search alongside the image
        candidate_masks.append(img_path.with_name(f"{stem}{mask_suffix}{img_path.suffix}"))

        # Search in provided mask dirs
        for mdir in mask_dirs_paths:
            candidate_masks.append(mdir / f"{stem}{mask_suffix}{img_path.suffix}")
            candidate_masks.append(mdir / f"{stem}{mask_suffix}.png")

        mask_path = next((p for p in candidate_masks if p.exists()), None)
        label = 0 if mask_path is not None else 1
        pairs.append((img_path, mask_path, label))
    return pairs


def write_jsonl(entries: List[Tuple[Path, Optional[Path], int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img, mask, label in entries:
            record = {"image": str(img.resolve()), "label": int(label)}
            if mask is not None:
                record["mask"] = str(mask.resolve())
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL manifests for document tamper dataset.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root of doctamper dataset")
    parser.add_argument("--mask-suffix", type=str, default="_mask", help="Suffix appended to stem for mask files")
    parser.add_argument("--mask-dirs", type=str, default="", help="Comma-separated subdirs to search for masks")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction for validation split")
    parser.add_argument("--train-out", type=Path, default=Path("data/train_manifest.jsonl"))
    parser.add_argument("--val-out", type=Path, default=Path("data/val_manifest.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    mask_dirs = [d for d in args.mask_dirs.split(",") if d]

    pairs = find_pairs(args.data_root, mask_suffix=args.mask_suffix, mask_dirs=mask_dirs)
    random.shuffle(pairs)

    split = int(len(pairs) * (1 - args.val_ratio))
    train_entries = pairs[:split]
    val_entries = pairs[split:]

    write_jsonl(train_entries, args.train_out)
    write_jsonl(val_entries, args.val_out)

    print(f"Wrote {len(train_entries)} train and {len(val_entries)} val entries.")


if __name__ == "__main__":
    main()
