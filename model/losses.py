"""Losses for single-band teacher distillation.

The positional CompressorLoss API matches the original Neural-Compressor-Script
training call. Curve tensors must already share the same frame grid; shape
mismatches are rejected instead of silently truncating misaligned curves.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompressorLoss", "curve_loss", "param_loss", "smoothness_loss",
           "W_STFT", "W_CURVE", "W_PARAM", "W_SMOOTH"]

# Preserve the original Neural-Compressor-Script objective weights.
W_STFT = 1.0
W_CURVE = 0.5
W_PARAM = 0.2
W_SMOOTH = 0.05


def _same_shape(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{label} shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")


def curve_loss(g_pred: torch.Tensor, g_tgt: torch.Tensor) -> torch.Tensor:
    _same_shape(g_pred, g_tgt, "gain curve")
    return F.l1_loss(g_pred, g_tgt)


def param_loss(p_pred: torch.Tensor, p_tgt: torch.Tensor) -> torch.Tensor:
    _same_shape(p_pred, p_tgt, "parameter")
    return F.l1_loss(p_pred, p_tgt)


def smoothness_loss(g_pred: torch.Tensor) -> torch.Tensor:
    if g_pred.ndim < 1:
        raise ValueError("gain curve must have at least one dimension")
    if g_pred.shape[-1] < 2:
        return g_pred.new_zeros(())
    return (g_pred[..., 1:] - g_pred[..., :-1]).abs().mean()


class _SpectralLoss(nn.Module):
    def __init__(self, n_fft: int, hop: int) -> None:
        super().__init__()
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _same_shape(pred, target, "audio")
        if pred.shape[-1] < self.n_fft:
            raise ValueError(f"audio length {pred.shape[-1]} is shorter than STFT window {self.n_fft}")
        a = torch.stft(pred.squeeze(1), self.n_fft, self.hop, self.n_fft,
                       window=self.window, center=True, return_complex=True).abs()
        b = torch.stft(target.squeeze(1), self.n_fft, self.hop, self.n_fft,
                       window=self.window, center=True, return_complex=True).abs()
        eps = 1e-7
        spectral_convergence = torch.linalg.vector_norm(a - b, dim=(-2, -1)) / (
            torch.linalg.vector_norm(b, dim=(-2, -1)) + eps
        )
        log_l1 = F.l1_loss(torch.log(a + eps), torch.log(b + eps))
        return spectral_convergence.mean() + log_l1


class CompressorLoss(nn.Module):
    def __init__(
        self, w_stft: float = W_STFT, w_curve: float = W_CURVE,
        w_param: float = W_PARAM, w_smooth: float = W_SMOOTH,
    ) -> None:
        super().__init__()
        self.w_stft = float(w_stft)
        self.w_curve = float(w_curve)
        self.w_param = float(w_param)
        self.w_smooth = float(w_smooth)
        self.spec_1024 = _SpectralLoss(1024, 256)
        self.spec_512 = _SpectralLoss(512, 128)

    def forward(
        self, y_pred: torch.Tensor, y_tgt: torch.Tensor,
        g_pred: torch.Tensor, g_tgt: torch.Tensor,
        p_pred: torch.Tensor, p_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _same_shape(y_pred, y_tgt, "audio")
        curve = curve_loss(g_pred, g_tgt)
        param = param_loss(p_pred, p_tgt)
        smooth = smoothness_loss(g_pred)
        stft = self.spec_1024(y_pred.float(), y_tgt.float()) + self.spec_512(
            y_pred.float(), y_tgt.float()
        )
        total = (self.w_stft * stft + self.w_curve * curve
                 + self.w_param * param + self.w_smooth * smooth)
        logs = {"total": total.detach(), "curve": curve.detach(),
                "stft": stft.detach(), "param": param.detach(),
                "smooth": smooth.detach()}
        return total, logs
