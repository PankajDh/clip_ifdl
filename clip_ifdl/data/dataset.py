import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ManifestEntry:
    image_path: str
    mask_path: Optional[str]
    label: int


def read_manifest(path: str) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            entries.append(
                ManifestEntry(
                    image_path=payload["image"],
                    mask_path=payload.get("mask"),
                    label=int(payload["label"]),
                )
            )
    return entries


class DocForgeryDataset(Dataset):
    """Dataset for business document tampering detection/localization."""

    def __init__(
        self,
        manifest_path: str,
        transform: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        image_size: int = 512,
    ) -> None:
        self.entries = read_manifest(manifest_path)
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image at {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, path: Optional[str]) -> Optional[np.ndarray]:
        if path is None:
            return None
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask at {path}")
        return mask

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = self._load_image(entry.image_path)
        mask = self._load_mask(entry.mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        label = torch.tensor(entry.label, dtype=torch.float32)
        return {"image": image, "mask": mask, "label": label}
