from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptAdjustmentNet(nn.Module):
    """Lightweight bottleneck to adapt prompts using the CLS token."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim // 16
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.net(cls_token)


class EmbeddingAdjustmentNet(nn.Module):
    """Transformer decoder to refine prompts with image tokens."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.alpha = nn.Parameter(torch.tensor(1e-3))

    def forward(self, prompts: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        refined = self.decoder(prompts, image_tokens)
        return prompts + self.alpha * refined


class InstanceAwareDualPrompt(nn.Module):
    """
    Implements the Instance-aware Dual-stream Prompt Learning (IDPL).

    We track separate prompt tokens for authentic and forged classes, initialize
    them from semantic seeds, adapt them with PANet using the CLS token, and
    refine them with EANet cross-attending image tokens.
    """

    def __init__(self, embed_dim: int, prompt_tokens: int = 8, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.prompt_tokens = prompt_tokens
        self.embed_dim = embed_dim

        # Prompt embeddings (N x D) for real/fake streams.
        self.real_prompt = nn.Parameter(torch.zeros(prompt_tokens, embed_dim))
        self.fake_prompt = nn.Parameter(torch.zeros(prompt_tokens, embed_dim))

        self.pan = PromptAdjustmentNet(embed_dim)
        self.ean = EmbeddingAdjustmentNet(embed_dim, heads=heads, dropout=dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.real_prompt, std=0.02)
        nn.init.normal_(self.fake_prompt, std=0.02)

    @torch.no_grad()
    def seed_from_text_embeddings(self, real_embed: torch.Tensor, fake_embed: torch.Tensor) -> None:
        """
        Initialize prompts from CLIP token embeddings derived from text seeds.
        Both tensors expected shape: (embed_dim,)
        """
        self.real_prompt.copy_(real_embed.unsqueeze(0).repeat(self.prompt_tokens, 1))
        self.fake_prompt.copy_(fake_embed.unsqueeze(0).repeat(self.prompt_tokens, 1))

    def _adjust(self, base_prompt: torch.Tensor, cls: torch.Tensor, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # base_prompt: (N, D)
        pan_shift = self.pan(cls)  # (B, D)
        prompt = base_prompt.unsqueeze(0) + pan_shift.unsqueeze(1)
        prompt = self.ean(prompt, image_tokens)
        pooled = prompt.mean(dim=1)
        return prompt, pooled

    def forward(self, cls_token: torch.Tensor, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_token: (B, D) token from CLIP visual encoder.
            image_tokens: (B, N, D) visual tokens (excluding CLS) from encoder.
        Returns:
            real_vec, fake_vec: pooled prompt embeddings.
            logits: similarity logits (B, 2) between CLS and prompts.
        """
        real_prompt, real_vec = self._adjust(self.real_prompt, cls_token, image_tokens)
        fake_prompt, fake_vec = self._adjust(self.fake_prompt, cls_token, image_tokens)

        cls_norm = F.normalize(cls_token, dim=-1)
        real_norm = F.normalize(real_vec, dim=-1)
        fake_norm = F.normalize(fake_vec, dim=-1)
        pos = (cls_norm * real_norm).sum(dim=-1)
        neg = (cls_norm * fake_norm).sum(dim=-1)
        logits = torch.stack([pos, neg], dim=-1)
        return real_vec, fake_vec, logits
