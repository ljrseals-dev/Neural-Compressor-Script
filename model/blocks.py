"""Transferred and adapted from ljrseals-dev/expert-lamp/model/blocks.py.

Adaptation: exposes the legacy Neural-Compressor-Script four-return API and latent
features for a separate ParameterProbe. Single-band teacher: n_bands must be 1.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompressorNet", "ParameterProbe", "build_model", "rms_envelope", "apply_gain_db"]

def rms_envelope(x: torch.Tensor, win: int = 512) -> torch.Tensor:
    win=max(2,int(win)); pad=win//2
    xp=F.pad(x,(pad,pad),mode="replicate")
    return torch.sqrt(F.avg_pool1d(xp*xp,win,stride=1).clamp_min(1e-12))

def apply_gain_db(x: torch.Tensor, g_db: torch.Tensor) -> torch.Tensor:
    return x*torch.pow(10.0,g_db/20.0)

class _ConvBlock(nn.Module):
    def __init__(self,ch:int,kernel:int,dilation:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(ch,ch,kernel,padding=dilation*(kernel//2),dilation=dilation),
            nn.GroupNorm(1,ch),nn.SiLU(),
            nn.Conv1d(ch,ch,kernel,padding=dilation*(kernel//2),dilation=dilation),
            nn.GroupNorm(1,ch))
    def forward(self,x): return F.silu(x+self.net(x))

class CompressorNet(nn.Module):
    """Small dilated CNN over waveform/envelope; predicts gain and latent features."""
    def __init__(self,channels=32,n_blocks=4,kernel=15,frame=256,env_win=512):
        super().__init__()
        self.frame=int(frame); self.env_win=int(env_win); self.latent_dim=int(channels)
        self.stem=nn.Conv1d(2,channels,kernel,padding=kernel//2)
        self.blocks=nn.ModuleList(_ConvBlock(channels,kernel,2**i) for i in range(n_blocks))
        self.curve_head=nn.Conv1d(channels,1,1)
        nn.init.zeros_(self.curve_head.weight); nn.init.zeros_(self.curve_head.bias)
    def count_parameters(self): return sum(p.numel() for p in self.parameters())
    def forward(self,x,return_extras=False):
        if x.ndim!=3 or x.shape[1]!=1: raise ValueError(f"expected mono (B,1,T), got {tuple(x.shape)}")
        T=x.shape[-1]
        env=rms_envelope(x,self.env_win)
        feat=torch.cat([x,torch.log10(env+1e-9)],dim=1)
        h=self.stem(feat)
        for b in self.blocks: h=b(h)
        g_full=self.curve_head(h).clamp(-60.0,0.0)
        g_p=F.avg_pool1d(g_full,self.frame,stride=self.frame,ceil_mode=True)
        g_full=F.interpolate(g_p,size=T,mode="linear",align_corners=False)
        y=apply_gain_db(x,g_full)
        latent=F.adaptive_avg_pool1d(h,1).squeeze(-1)
        extras={"latent":latent,"y":y,"g_full":g_full}
        # Single-band output only; a second band would have no teacher target.
        g_bands=g_p.unsqueeze(1)
        if return_extras: return y,g_p,g_bands,extras
        return y,g_p,g_bands,extras

class ParameterProbe(nn.Module):
    def __init__(self,latent_dim:int,n_params:int=4):
        super().__init__()
        if n_params!=4: raise ValueError("compressor probe requires exactly four parameters")
        self.net=nn.Sequential(nn.Linear(latent_dim,64),nn.SiLU(),nn.Linear(64,4),nn.Sigmoid())
    def forward(self,latent): return self.net(latent)

def build_model(n_bands:int=1):
    if int(n_bands)!=1: raise ValueError("Transferred expert-lamp teacher is single-band; n_bands must be 1.")
    return CompressorNet()
