from typing import Optional, Tuple

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def _resize_keep_aspect(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_h, pad_w = target - nh, target - nw
    top = pad_h // 2
    left = pad_w // 2
    bottom = pad_h - top
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded


def default_transform(image: np.ndarray, mask: Optional[np.ndarray], image_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    image = _resize_keep_aspect(image, image_size)
    image = image.astype("float32") / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = torch.from_numpy(image).permute(2, 0, 1)

    mask_tensor: Optional[torch.Tensor] = None
    if mask is not None:
        mask = _resize_keep_aspect(mask, image_size)
        mask = (mask > 127).astype("float32")
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
    return image, mask_tensor
