from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv(nn.Module):
    """Constrained convolution to highlight manipulation noise [Bayar & Stamm]."""

    def __init__(self, channels: int = 3, kernel_size: int = 5) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        weight = torch.zeros((3, channels, kernel_size, kernel_size))
        nn.init.normal_(weight, mean=0.0, std=0.01)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # Project weights to satisfy Bayar constraint: center=-1, others sum to 1.
        w_proj = w.clone()
        w_proj[:, :, self.pad, self.pad] = 0.0
        norm = w_proj.sum(dim=(2, 3), keepdim=True) + 1e-6
        w_proj = w_proj / norm
        w_proj[:, :, self.pad, self.pad] = -1.0
        return F.conv2d(x, w_proj, padding=self.pad)


class NoiseFeatureExtractor(nn.Module):
    """Extracts noise-domain cues and aligns them to CLIP's embedding width."""

    def __init__(self, embed_dim: int, noise_channels: int = 64, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.bayar = BayarConv()
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(3, noise_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, noise_channels),
            nn.GELU(),
            nn.Conv2d(noise_channels, noise_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, noise_channels),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(noise_channels * 2, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid half-complex ops: disable autocast inside this block.
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()

            # RGB -> noise residuals
            noise = self.bayar(x_fp32)
            noise_feat = self.noise_encoder(noise)

            # Frequency magnitude as complementary domain
            freq = torch.fft.rfft2(noise, norm="ortho")
            freq_mag = torch.abs(freq)
            freq_mag = torch.fft.irfft2(freq_mag, s=noise.shape[-2:], norm="ortho")
            freq_feat = self.noise_encoder(freq_mag)

            fused = torch.cat([noise_feat, freq_feat], dim=1)
            fused = self.fusion(fused)

            if fused.shape[-1] % self.patch_size != 0 or fused.shape[-2] % self.patch_size != 0:
                raise ValueError("Input resolution must be divisible by patch size for noise adapter.")
            tokens = F.unfold(fused, kernel_size=self.patch_size, stride=self.patch_size)
            tokens = tokens.transpose(1, 2)
            patch_area = self.patch_size * self.patch_size
            tokens = tokens.view(tokens.shape[0], tokens.shape[1], fused.shape[1], patch_area).mean(dim=-1)
        return tokens.to(x.dtype)


class NoiseAdapterBlock(nn.Module):
    """Cross-attends CLIP tokens with noise tokens using zero-init gates."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Start with zeroed projections to avoid hurting frozen CLIP at init.
        for name, param in self.attn.named_parameters():
            nn.init.zeros_(param)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, noise_tokens: torch.Tensor) -> torch.Tensor:
        # tokens: B x (1+N) x D, noise_tokens: B x N x D
        attn_out, _ = self.attn(tokens, noise_tokens, noise_tokens, need_weights=False)
        tokens = tokens + self.gate * attn_out
        tokens = tokens + self.gate * self.ffn(tokens)
        return tokens


class ForgeryEnhancedNoiseAdapter(nn.Module):
    """
    Injects noise-domain cues at selected transformer layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        adapter_layers: Iterable[int],
        patch_size: int,
        noise_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adapter_layers: List[int] = list(adapter_layers)
        self.noise_extractor = NoiseFeatureExtractor(embed_dim, noise_channels=noise_channels, patch_size=patch_size)
        self.adapters = nn.ModuleDict(
            {str(idx): NoiseAdapterBlock(embed_dim, num_heads=num_heads, dropout=dropout) for idx in self.adapter_layers}
        )

    def forward_noise(self, image: torch.Tensor) -> torch.Tensor:
        return self.noise_extractor(image)

    def apply(self, layer_idx: int, tokens: torch.Tensor, noise_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if noise_tokens is None:
            return tokens
        if layer_idx not in self.adapter_layers:
            return tokens
        adapter = self.adapters[str(layer_idx)]
        return adapter(tokens, noise_tokens)
