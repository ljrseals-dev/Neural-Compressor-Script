"""Single-band neural compressor recovered from expert-lamp.

Training contract:
    build_model(n_bands=1)(x, return_extras=True)
        -> (y_pred, g_pred, g_bands, extras)
    extras["latent"] is consumed by ParameterProbe.

The teacher is single-band. n_bands values other than one are rejected rather
than fabricating multi-band targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompressorNet", "ParameterProbe", "build_model", "rms_envelope", "apply_gain_db"]


def rms_envelope(x: torch.Tensor, win: int = 512) -> torch.Tensor:
    win = max(2, int(win))
    left = win // 2
    right = win - 1 - left
    xp = F.pad(x, (left, right), mode="replicate")
    return torch.sqrt(F.avg_pool1d(xp * xp, win, stride=1).clamp_min(1e-12))


def apply_gain_db(x: torch.Tensor, g_db: torch.Tensor) -> torch.Tensor:
    return x * torch.pow(10.0, g_db / 20.0)


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=dilation * (kernel // 2), dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel, padding=dilation * (kernel // 2), dilation=dilation),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class CompressorNet(nn.Module):
    """Dilated waveform/envelope CNN with one gain-curve output and a latent."""

    def __init__(
        self, channels: int = 32, n_blocks: int = 4, kernel: int = 15,
        frame: int = 256, env_win: int = 512,
    ) -> None:
        super().__init__()
        if frame < 1:
            raise ValueError("frame must be >= 1")
        self.frame = int(frame)
        self.env_win = int(env_win)
        self.stem = nn.Conv1d(2, channels, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList(
            _ConvBlock(channels, kernel, dilation=2 ** i) for i in range(n_blocks)
        )
        self.curve_head = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.curve_head.weight)
        nn.init.zeros_(self.curve_head.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, return_extras: bool = True):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected mono input (B,1,T), got {tuple(x.shape)}")
        if x.shape[-1] < self.frame:
            raise ValueError(f"audio length must be >= frame ({self.frame})")
        t = x.shape[-1]
        env = rms_envelope(x, self.env_win)
        features = torch.cat((x, torch.log10(env + 1e-9)), dim=1)
        h = self.stem(features)
        for block in self.blocks:
            h = block(h)
        latent = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        g_sample = self.curve_head(h).clamp(-60.0, 0.0)
        # The prediction and teacher target both use frame-sized means,
        # including the same ceil-mode treatment of the final partial frame.
        g_pred = F.avg_pool1d(g_sample, self.frame, stride=self.frame, ceil_mode=True)
        g_full = F.interpolate(g_pred, size=t, mode="linear", align_corners=False)
        y_pred = apply_gain_db(x, g_full)
        g_bands = g_pred.unsqueeze(1)  # (B, 1 band, 1, frames)
        extras = {"latent": latent, "g_full": g_full}
        return y_pred, g_pred, g_bands, extras


class ParameterProbe(nn.Module):
    """Regress normalized [threshold, ratio, attack, release] from latent features."""

    def __init__(self, latent_dim: int, n_params: int = 4) -> None:
        super().__init__()
        if latent_dim < 1 or n_params != 4:
            raise ValueError("latent_dim must be positive and n_params must equal 4")
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_params),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def build_model(n_bands: int = 1, **kwargs) -> CompressorNet:
    if int(n_bands) != 1:
        raise ValueError("this recovered teacher/model contract is single-band; use n_bands=1")
    return CompressorNet(**kwargs)
