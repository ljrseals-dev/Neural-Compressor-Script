"""Synthetic data and single-band teacher transferred from expert-lamp.

Adaptations: legacy dataset API, optional precompute, collate helper, and
teacher gain resampling onto the student's 256-sample pooling grid.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
PARAM_RANGES={"threshold_db":(-60.0,0.0),"ratio":(1.0,20.0),"attack_ms":(0.1,200.0),"release_ms":(10.0,2000.0)}
def denormalize_params(p):
    thr=p[...,0]*60.0-60.0
    ratio=torch.exp(torch.log(torch.tensor(1.0))+p[...,1]*(math.log(20.0)-math.log(1.0)))
    att=torch.exp(torch.log(torch.tensor(0.1))+p[...,2]*(math.log(200.0)-math.log(0.1)))
    rel=torch.exp(torch.log(torch.tensor(10.0))+p[...,3]*(math.log(2000.0)-math.log(10.0)))
    return {"threshold_db":thr,"ratio":ratio,"attack_ms":att,"release_ms":rel}
@torch.no_grad()
def teacher_compress(x,sample_rate,params,*,detect_sr=1000,knee_db=6.0):
    B,T=x.shape; p=denormalize_params(params)
    thr=p["threshold_db"].to(x.dtype); ratio=p["ratio"].to(x.dtype)
    aa=torch.exp(-1.0/(p["attack_ms"]*1e-3*detect_sr)).to(x.dtype)
    ar=torch.exp(-1.0/(p["release_ms"]*1e-3*detect_sr)).to(x.dtype)
    hop=max(1,sample_rate//detect_sr); n=T//hop
    xr=x[:,:n*hop].reshape(B,n,hop); env=torch.sqrt((xr*xr).mean(-1).clamp_min(1e-12))
    env_db=20.0*torch.log10(env+1e-9); over=env_db-thr.unsqueeze(1); slope=1.0-1.0/ratio.unsqueeze(1)
    gr=torch.where(over<=-knee_db/2,torch.zeros_like(over),torch.where(over>=knee_db/2,-slope*over,-slope*(over+knee_db/2)**2/(2*knee_db)))
    out=torch.empty_like(gr); prev=gr[:,0]
    for i in range(n):
        tgt=gr[:,i]; a=torch.where(tgt<prev,aa,ar); prev=a*prev+(1.0-a)*tgt; out[:,i]=prev
    gfull=F.interpolate(out.unsqueeze(1),size=T,mode="linear",align_corners=False).squeeze(1)
    y=x*torch.pow(10.0,gfull/20.0)
    return y,out
class SyntheticCompressorDataset(Dataset):
    def __init__(self,n_samples=256,duration_s=1.5,sample_rate=44100,precompute=True,seed=0,detect_sr=1000,frame=256):
        self.n=int(n_samples); self.T=max(256,int(round(duration_s*sample_rate))); self.sr=int(sample_rate); self.seed=int(seed); self.detect_sr=int(detect_sr); self.frame=int(frame); self.precompute=bool(precompute)
        self.cache=[self._make(i) for i in range(self.n)] if self.precompute else None
    def __len__(self): return self.n
    def _signal(self,g):
        T,sr=self.T,self.sr; t=torch.arange(T,dtype=torch.float32)/sr
        f0=60.0*torch.pow(2.0,torch.rand((),generator=g)*4.0)
        tone=torch.sin(2*math.pi*f0*t)*torch.exp(-t*torch.rand((),generator=g)*3.0); tone=tone*(0.3+0.7*torch.rand((),generator=g))
        burst=torch.randn(T,generator=g)*torch.exp(-t*(5.0+15.0*torch.rand((),generator=g)))*0.5
        bed=torch.randn(T,generator=g)*0.05*torch.exp(-t*0.5*torch.rand((),generator=g))
        gesture=0.5+0.5*torch.sin(2*math.pi*(0.2+2.0*torch.rand((),generator=g))*t+torch.rand((),generator=g)*6.28)
        x=(tone+burst)*gesture+bed
        return (x/(x.abs().max()+1e-6)*0.9).clamp(-1.0,1.0)
    def _params(self,g):
        return torch.stack([torch.rand((),generator=g)*0.8+0.1,torch.rand((),generator=g)*0.7+0.1,torch.rand((),generator=g)*0.6+0.1,torch.rand((),generator=g)*0.8+0.1])
    def _make(self,idx):
        g=torch.Generator().manual_seed(self.seed*1_000_003+idx); x=self._signal(g).unsqueeze(0); p=self._params(g)
        y,gt_coarse=teacher_compress(x,self.sr,p.unsqueeze(0),detect_sr=self.detect_sr)
        gt_full=F.interpolate(gt_coarse.unsqueeze(1),size=self.T,mode="linear",align_corners=False)
        gt_frames=F.avg_pool1d(gt_full,self.frame,stride=self.frame,ceil_mode=True)
        return {"x":x,"y":y,"g":gt_frames.squeeze(0),"params":p,"idx":torch.tensor(idx)}
    def __getitem__(self,idx):
        return self.cache[idx] if self.cache is not None else self._make(int(idx))
def collate(samples):
    if not samples: raise ValueError("cannot collate empty sample list")
    return {k:torch.stack([s[k] for s in samples]) for k in ("x","y","g","params","idx")}
