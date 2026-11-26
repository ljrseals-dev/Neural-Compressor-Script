"""Deterministic synthetic audio dataset and classical compressor teacher.

Recovered from expert-lamp. The teacher detector runs at a coarse hop grid,
but its gain target is explicitly projected onto the student's 256-sample
frame grid before it is returned. No array truncation is used as alignment.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["CompressorDataset", "SyntheticCompressorDataset", "teacher_compress",
           "denormalize_params", "collate", "PARAM_RANGES"]

PARAM_RANGES = {
    "threshold_db": (-60.0, 0.0),
    "ratio": (1.0, 20.0),
    "attack_ms": (0.1, 200.0),
    "release_ms": (10.0, 2000.0),
}


def denormalize_params(p: torch.Tensor) -> dict[str, torch.Tensor]:
    if p.shape[-1] != 4:
        raise ValueError("params must end in four normalized values")
    device, dtype = p.device, p.dtype
    thr = p[..., 0] * 60.0 - 60.0
    ratio = torch.exp(torch.log(torch.tensor(1.0, device=device, dtype=dtype))
                      + p[..., 1] * math.log(20.0))
    attack = torch.exp(torch.log(torch.tensor(0.1, device=device, dtype=dtype))
                       + p[..., 2] * (math.log(200.0) - math.log(0.1)))
    release = torch.exp(torch.log(torch.tensor(10.0, device=device, dtype=dtype))
                        + p[..., 3] * (math.log(2000.0) - math.log(10.0)))
    return {"threshold_db": thr, "ratio": ratio, "attack_ms": attack, "release_ms": release}


@torch.no_grad()
def teacher_compress(
    x: torch.Tensor, sample_rate: int, params: torch.Tensor,
    *, detect_sr: int = 1000, knee_db: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or params.ndim != 2 or params.shape[-1] != 4:
        raise ValueError("expected x=(B,T) and params=(B,4)")
    if sample_rate < 1 or detect_sr < 1 or x.shape[-1] < 1:
        raise ValueError("sample_rate, detect_sr and audio length must be positive")
    b, length = x.shape
    p = denormalize_params(params)
    thr, ratio = p["threshold_db"].to(x.dtype), p["ratio"].to(x.dtype)
    a_att = torch.exp(-1.0 / (p["attack_ms"] * 1e-3 * detect_sr)).to(x.dtype)
    a_rel = torch.exp(-1.0 / (p["release_ms"] * 1e-3 * detect_sr)).to(x.dtype)
    hop = max(1, sample_rate // detect_sr)
    frames = math.ceil(length / hop)
    padded = F.pad(x, (0, frames * hop - length))
    xr = padded.reshape(b, frames, hop)
    env = torch.sqrt((xr * xr).mean(-1).clamp_min(1e-12))
    env_db = 20.0 * torch.log10(env + 1e-9)
    over = env_db - thr.unsqueeze(1)
    slope = 1.0 - 1.0 / ratio.unsqueeze(1)
    knee = float(knee_db)
    gr = torch.where(
        over <= -knee / 2, torch.zeros_like(over),
        torch.where(over >= knee / 2, -slope * over,
                    -slope * (over + knee / 2).square() / (2.0 * knee)),
    )
    out = torch.empty_like(gr)
    prev = gr[:, 0]
    for i in range(frames):
        target = gr[:, i]
        coeff = torch.where(target < prev, a_att, a_rel)
        prev = coeff * prev + (1.0 - coeff) * target
        out[:, i] = prev
    gain_full = F.interpolate(out.unsqueeze(1), size=length, mode="linear",
                              align_corners=False).squeeze(1)
    y = x * torch.pow(10.0, gain_full / 20.0)
    return y, gain_full


class CompressorDataset(Dataset):
    def __init__(
        self, n_samples: int = 256, duration_s: float = 1.5,
        sample_rate: int = 44100, seed: int = 0, detect_sr: int = 1000,
        frame: int = 256, precompute: bool = False,
    ) -> None:
        if n_samples < 1 or duration_s <= 0 or sample_rate < 1 or frame < 1:
            raise ValueError("n_samples, duration, sample_rate and frame must be positive")
        self.n = int(n_samples)
        self.T = int(round(duration_s * sample_rate))
        if self.T < frame:
            raise ValueError(f"duration produces {self.T} samples; model requires at least {frame}")
        self.sr, self.seed, self.detect_sr, self.frame = int(sample_rate), int(seed), int(detect_sr), int(frame)
        self.precompute = bool(precompute)
        self._cache = [self._make_item(i) for i in range(self.n)] if self.precompute else None

    def __len__(self) -> int:
        return self.n

    def _signal(self, generator: torch.Generator) -> torch.Tensor:
        t = torch.arange(self.T, dtype=torch.float32) / self.sr
        f0 = 60.0 * torch.pow(2.0, torch.rand((), generator=generator) * 4.0)
        tone = torch.sin(2 * math.pi * f0 * t) * torch.exp(
            -t * torch.rand((), generator=generator) * 3.0)
        tone = tone * (0.3 + 0.7 * torch.rand((), generator=generator))
        burst = torch.randn(self.T, generator=generator) * torch.exp(
            -t * (5.0 + 15.0 * torch.rand((), generator=generator))) * 0.5
        bed = torch.randn(self.T, generator=generator) * 0.05 * torch.exp(
            -t * 0.5 * torch.rand((), generator=generator))
        gesture = 0.5 + 0.5 * torch.sin(
            2 * math.pi * (0.2 + 2.0 * torch.rand((), generator=generator)) * t
            + torch.rand((), generator=generator) * 6.28)
        x = (tone + burst) * gesture + bed
        return (x / (x.abs().max() + 1e-6) * 0.9).clamp(-1.0, 1.0)

    def _make_item(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        x = self._signal(generator).unsqueeze(0)
        params = torch.stack((
            torch.rand((), generator=generator) * 0.8 + 0.1,
            torch.rand((), generator=generator) * 0.7 + 0.1,
            torch.rand((), generator=generator) * 0.6 + 0.1,
            torch.rand((), generator=generator) * 0.8 + 0.1,
        ))
        y, gain_full = teacher_compress(x, self.sr, params.unsqueeze(0),
                                        detect_sr=self.detect_sr)
        # Exactly matches CompressorNet's avg_pool1d(kernel=stride=frame,
        # ceil_mode=True) grid, including the final partial frame.
        gain_frames = F.avg_pool1d(gain_full.unsqueeze(1), self.frame,
                                   stride=self.frame, ceil_mode=True)
        return {
            "x": x, "y": y, "g": gain_frames.squeeze(0),
            "params": params, "idx": torch.tensor(idx, dtype=torch.long),
            "sample_id": torch.tensor([self.seed, idx], dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.n
        if idx < 0 or idx >= self.n:
            raise IndexError(idx)
        if self._cache is not None:
            return {k: v.clone() for k, v in self._cache[idx].items()}
        return self._make_item(idx)


class SyntheticCompressorDataset(CompressorDataset):
    """Compatibility name for the original Neural-Compressor-Script import."""
    pass


def collate(samples: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    samples = list(samples)
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = samples[0].keys()
    if any(sample.keys() != samples[0].keys() for sample in samples):
        raise ValueError("all samples must expose identical keys")
    return {key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in keys}
