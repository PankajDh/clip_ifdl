import math
from typing import Dict, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fena import ForgeryEnhancedNoiseAdapter
from .idpl import InstanceAwareDualPrompt


class LocalizationHead(nn.Module):
    """Lightweight decoder that upsamples patch tokens to a full-resolution mask."""

    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, image_size: int) -> torch.Tensor:
        # tokens: B x N x D (N = H_p * W_p)
        b, n, _ = tokens.shape
        grid = int(math.sqrt(n))
        mask = self.proj(tokens)
        mask = mask.transpose(1, 2).reshape(b, 1, grid, grid)
        mask = F.interpolate(mask, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return mask


class CLIPIFDL(nn.Module):
    """
    PyTorch implementation of CLIP-IFDL.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        prompt_tokens: int = 8,
        adapter_heads: int = 8,
        adapter_layers: Optional[list] = None,
        noise_channels: int = 64,
        dropout: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if adapter_layers is None:
            adapter_layers = [0, 3, 6, 9, 12]

        self.model_name = model_name
        self.pretrained = pretrained

        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip = clip_model
        self.visual = clip_model.visual
        self.width = getattr(self.visual, "width", clip_model.text_projection.shape[0])
        self.patch_size = self.visual.patch_size if isinstance(self.visual.patch_size, int) else self.visual.patch_size[0]
        self.device = device

        self.fena = ForgeryEnhancedNoiseAdapter(
            embed_dim=self.width,
            num_heads=adapter_heads,
            adapter_layers=adapter_layers,
            patch_size=self.patch_size,
            noise_channels=noise_channels,
            dropout=dropout,
        )
        self.idpl = InstanceAwareDualPrompt(
            embed_dim=self.width, prompt_tokens=prompt_tokens, heads=adapter_heads, dropout=dropout
        )
        self.loc_head = LocalizationHead(embed_dim=self.width, patch_size=self.patch_size)

        self._seed_prompts()

    @torch.no_grad()
    def _seed_prompts(self) -> None:
        tokenizer = open_clip.get_tokenizer(self.model_name)
        real_tokens = tokenizer("a real document.")
        fake_tokens = tokenizer("a forged document.")
        # Derive a single embedding per phrase by averaging non-padding tokens.
        def _phrase_embedding(token_ids: torch.Tensor) -> torch.Tensor:
            token_ids = token_ids.to(self.device)
            token_embeds = self.clip.token_embedding(token_ids)  # 1 x 77 x D
            mask = token_ids.ne(0).float().unsqueeze(-1)
            pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled.squeeze(0)

        real_embed = _phrase_embedding(real_tokens)
        fake_embed = _phrase_embedding(fake_tokens)
        self.idpl.seed_from_text_embeddings(real_embed.cpu(), fake_embed.cpu())

    def _visual_forward(self, image: torch.Tensor, noise_tokens: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Adapted from open_clip VisionTransformer forward with adapter hooks.
        x = self.visual.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # B, N, D

        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype
        )
        x = torch.cat([cls_token, x], dim=1)

        # Resize positional embedding to match token count if needed.
        pos_embed = self.visual.positional_embedding.to(x.dtype)
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed = self._resize_pos_embed(pos_embed, target_seq=x.shape[1])
        x = x + pos_embed
        x = self.visual.ln_pre(x)

        tokens = x
        # Optional adapter before the first transformer layer.
        tokens = self.fena.apply(0, tokens, noise_tokens)

        x = tokens.permute(1, 0, 2)  # seq, batch, dim for CLIP blocks
        for idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            x = block(x)
            tokens = x.permute(1, 0, 2)
            tokens = self.fena.apply(idx, tokens, noise_tokens)
            x = tokens.permute(1, 0, 2)

        # Adapter after the final layer if requested (e.g., layer idx == 12).
        tokens = x.permute(1, 0, 2)
        tokens = self.fena.apply(len(self.visual.transformer.resblocks), tokens, noise_tokens)
        x = tokens.permute(1, 0, 2)

        x = x.permute(1, 0, 2)  # B, seq, dim
        x = self.visual.ln_post(x)
        cls = x[:, 0, :]
        tokens = x[:, 1:, :]
        return {"cls": cls, "tokens": tokens}

    def _resize_pos_embed(self, pos_embed: torch.Tensor, target_seq: int) -> torch.Tensor:
        """Interpolate 2D grid positional embedding to match new sequence length."""
        # pos_embed: (1, 1 + Gh*Gw, D)
        cls = pos_embed[:, :1, :]
        tok = pos_embed[:, 1:, :]
        old_tokens = tok.shape[1]
        old_size = int(math.sqrt(old_tokens))
        new_tokens = target_seq - 1
        new_size = int(math.sqrt(new_tokens))
        tok = tok.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        tok = F.interpolate(tok, size=(new_size, new_size), mode="bicubic", align_corners=False)
        tok = tok.permute(0, 2, 3, 1).reshape(1, new_tokens, -1)
        return torch.cat([cls, tok], dim=1)

    def forward(self, image: torch.Tensor, image_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if image_size is None:
            image_size = image.shape[-1]
        noise_tokens = self.fena.forward_noise(image)
        visual_out = self._visual_forward(image, noise_tokens)
        cls = visual_out["cls"]
        tokens = visual_out["tokens"]

        _, _, logits = self.idpl(cls, tokens)
        det_logit = logits[:, 0] - logits[:, 1]  # positive when leaning toward "real"

        loc_logits = self.loc_head(tokens, image_size=image_size)

        return {"cls": cls, "tokens": tokens, "det_logit": det_logit, "loc_logits": loc_logits}
