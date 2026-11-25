"""Losses adapted from expert-lamp for the original training call contract.

Gain curves must already be on the same frame grid. Dataset adapter performs
teacher/student grid alignment. STFT loss is computed on waveform outputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

W_STFT,W_CURVE,W_PARAM,W_SMOOTH=1.0,0.5,0.2,0.05

def curve_loss(g_p,g_t):
    if g_p.shape!=g_t.shape: raise ValueError(f"unaligned gain grids: {tuple(g_p.shape)} vs {tuple(g_t.shape)}")
    return F.l1_loss(g_p,g_t)

def param_loss(p_pred,p_t): return F.l1_loss(p_pred,p_t)

def smoothness_loss(g):
    if g.shape[-1]<2: return g.new_zeros(())
    return F.l1_loss(g[...,1:]-g[...,:-1],torch.zeros_like(g[...,1:]))

class _SpectralLoss(nn.Module):
    def __init__(self,n_fft,hop):
        super().__init__(); self.n_fft=n_fft; self.hop=hop
        self.register_buffer("window",torch.hann_window(n_fft),persistent=False)
    def forward(self,a,b):
        n=min(a.shape[-1],b.shape[-1]); a=a[...,:n]; b=b[...,:n]
        A=torch.stft(a.squeeze(1),self.n_fft,self.hop,self.n_fft,window=self.window,center=True,return_complex=True).abs()
        B=torch.stft(b.squeeze(1),self.n_fft,self.hop,self.n_fft,window=self.window,center=True,return_complex=True).abs()
        eps=1e-7
        sc=torch.linalg.vector_norm(A-B,dim=-1)/(torch.linalg.vector_norm(B,dim=-1)+eps)
        return sc.mean()+F.l1_loss(torch.log(A+eps),torch.log(B+eps))

class CompressorLoss(nn.Module):
    def __init__(self,w_stft=W_STFT,w_curve=W_CURVE,w_param=W_PARAM,w_smooth=W_SMOOTH):
        super().__init__(); self.weights=(float(w_stft),float(w_curve),float(w_param),float(w_smooth))
        self.spec1=_SpectralLoss(1024,256); self.spec2=_SpectralLoss(512,128)
    def forward(self,y_p,y_t,g_p,g_t,p_pred,p_t):
        curve=curve_loss(g_p.float(),g_t.float()); param=param_loss(p_pred.float(),p_t.float())
        smooth=smoothness_loss(g_p.float())
        stft=self.spec1(y_p.float(),y_t.float())+self.spec2(y_p.float(),y_t.float())
        total=self.weights[0]*stft+self.weights[1]*curve+self.weights[2]*param+self.weights[3]*smooth
        logs={"total":total,"stft":stft.detach(),"curve":curve.detach(),"param":param.detach(),"smooth":smooth.detach()}
        return total,logs
