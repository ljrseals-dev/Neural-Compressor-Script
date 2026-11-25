#!/usr/bin/env python3
"""Train the recovered single-band neural compressor by teacher distillation."""
from __future__ import annotations
import argparse, json, math, platform, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.blocks import ParameterProbe, build_model
from model.losses import (W_CURVE, W_PARAM, W_SMOOTH, W_STFT, CompressorLoss,
                          curve_loss, param_loss, smoothness_loss)
from training.dataset import SyntheticCompressorDataset, collate

CHECKPOINT_FORMAT_VERSION = 1
DATASET_CONTRACT_VERSION = "expert-lamp-teacher-v1-aligned-frame256"
FRAME_SIZE = 256


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic: torch.use_deterministic_algorithms(True, warn_only=True)


def safe_torch_load(path: Path, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch must support weights_only=True; unrestricted pickle loading is disabled") from exc


def make_scaler(device, enabled):
    active = bool(enabled and device.type == "cuda")
    try: return torch.amp.GradScaler(device.type, enabled=active)
    except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=active)


def amp_context(device, enabled):
    try: return torch.autocast(device_type=device.type, enabled=enabled)
    except (AttributeError, TypeError): return torch.cuda.amp.autocast(enabled=enabled)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stft-every", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-bands", type=int, default=1, help="Only one band is supported by this teacher.")
    p.add_argument("--no-amp", action="store_false", dest="amp", default=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict-semantic", action="store_true",
                   help="Deprecated; semantic_postcheck is absent.")
    return p.parse_args(argv)


def model_outputs(model, probe, x):
    y, g, bands, extras = model(x, return_extras=True)
    if not isinstance(extras, dict) or "latent" not in extras:
        raise TypeError("model extras must contain latent")
    return y, g, bands, probe(extras["latent"])


def make_loader(ds, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=workers, drop_last=False, generator=generator)


def check_disjoint(*datasets):
    ids = [{(int(ds.seed), i) for i in range(len(ds))} for ds in datasets]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] & ids[j]: raise RuntimeError("train/validation/test sample IDs overlap")


def batch_loss(criterion, yp, yt, gp, gt, pp, pt, use_stft):
    if use_stft:
        return criterion(yp.float(), yt.float(), gp.float(), gt.float(), pp.float(), pt.float())
    c, p, s = curve_loss(gp.float(), gt.float()), param_loss(pp.float(), pt.float()), smoothness_loss(gp.float())
    total = W_CURVE*c + W_PARAM*p + W_SMOOTH*s
    return total, {"total": total.detach(), "curve": c.detach(),
                   "stft": total.detach().new_zeros(()), "param": p.detach(), "smooth": s.detach()}


def evaluate(model, probe, loader, criterion, device):
    model.eval(); probe.eval()
    sums, counts = {}, {}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            yp, gp, _, pp = model_outputs(model, probe, x)
            total, logs = batch_loss(criterion, yp, batch["y"].to(device), gp,
                                     batch["g"].to(device), pp, batch["params"].to(device), True)
            if not torch.isfinite(total): raise FloatingPointError("non-finite evaluation objective")
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v); counts[k] = counts.get(k, 0) + 1
    model.train(); probe.train()
    return {k: sums[k] / counts[k] for k in sums}


def config_for(args):
    return {"sample_rate": args.sample_rate, "duration": args.duration, "n_bands": 1,
            "frame": FRAME_SIZE, "dataset_contract": DATASET_CONTRACT_VERSION,
            "seed": args.seed, "n_samples": args.n_samples, "n_val": args.n_val,
            "n_test": args.n_test, "batch_size": args.batch_size, "lr": args.lr,
            "stft_every": args.stft_every, "grad_clip": args.grad_clip,
            "loss_weights": {"stft": W_STFT, "curve": W_CURVE, "param": W_PARAM, "smooth": W_SMOOTH}}


def save_checkpoint(path, model, probe, optimizer, scheduler, scaler, epoch, best, history, args):
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "config": config_for(args),
                "model": model.state_dict(), "probe": probe.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "epoch": int(epoch),
                "best_val_curve": float(best), "history": history}, path)


def load_checkpoint(path, model, probe, optimizer, scheduler, scaler, args):
    ck = safe_torch_load(path)
    required = {"format_version","config","model","probe","optimizer","scheduler",
                "scaler","epoch","best_val_curve","history"}
    if not isinstance(ck, dict) or required - set(ck):
        raise RuntimeError(f"checkpoint missing keys: {sorted(required - set(ck)) if isinstance(ck, dict) else sorted(required)}")
    if ck["format_version"] != CHECKPOINT_FORMAT_VERSION: raise RuntimeError("unsupported checkpoint format")
    if ck["config"] != config_for(args): raise RuntimeError("checkpoint config mismatch; explicit migration required")
    model.load_state_dict(ck["model"], strict=True); probe.load_state_dict(ck["probe"], strict=True)
    optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    if not isinstance(ck["history"], list): raise RuntimeError("checkpoint history must be a list")
    return int(ck["epoch"]) + 1, float(ck["best_val_curve"]), ck["history"]


def main(argv=None):
    args = parse_args(argv)
    if args.epochs < 1 or min(args.n_samples,args.n_val,args.n_test,args.batch_size,args.sample_rate) < 1:
        raise SystemExit("epochs, sample counts, batch size and sample rate must be positive")
    if args.duration * args.sample_rate < 1024: raise SystemExit("duration must provide at least 1024 samples for STFT")
    if args.n_bands != 1: raise SystemExit("only --n-bands 1 is supported by the recovered single-band teacher")
    if args.strict_semantic: raise SystemExit("--strict-semantic unsupported: harmonia_neural.semantic_postcheck is absent")
    if args.stft_every < 1 or args.grad_clip < 0: raise SystemExit("--stft-every must be >=1 and --grad-clip >=0")
    args.out_dir.mkdir(parents=True, exist_ok=True); set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    train_ds = SyntheticCompressorDataset(args.n_samples,args.duration,args.sample_rate,args.seed,frame=FRAME_SIZE,precompute=True)
    val_ds = SyntheticCompressorDataset(args.n_val,args.duration,args.sample_rate,args.seed+7919,frame=FRAME_SIZE,precompute=True)
    test_ds = SyntheticCompressorDataset(args.n_test,args.duration,args.sample_rate,args.seed+15839,frame=FRAME_SIZE,precompute=True)
    check_disjoint(train_ds,val_ds,test_ds)
    train_loader = make_loader(train_ds,args.batch_size,True,args.num_workers,args.seed)
    val_loader = make_loader(val_ds,args.batch_size,False,0,args.seed+1)
    test_loader = make_loader(test_ds,args.batch_size,False,0,args.seed+2)
    model = build_model(n_bands=1).to(device)
    probe = ParameterProbe(model.stem.out_channels,4).to(device)
    criterion = CompressorLoss().to(device)
    params = list(model.parameters()) + list(probe.parameters())
    optimizer = torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler = make_scaler(device,args.amp)
    if args.dry_run:
        b = next(iter(train_loader))
        with torch.no_grad(): yp,g,bands,pp = model_outputs(model,probe,b["x"].to(device))
        print({"x":tuple(b["x"].shape),"y":tuple(yp.shape),"g":tuple(g.shape),
               "g_bands":tuple(bands.shape),"params":tuple(pp.shape),
               "finite":bool(torch.isfinite(yp).all())})
        return 0
    start,best,history = 0,math.inf,[]
    if args.resume is not None:
        start,best,history = load_checkpoint(args.resume,model,probe,optimizer,scheduler,scaler,args)
    started = time.time()
    for epoch in range(start,args.epochs):
        model.train(); probe.train(); sums,counts = {},{}
        for step,b in enumerate(train_loader):
            x,yt,gt,pt = b["x"].to(device),b["y"].to(device),b["g"].to(device),b["params"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device,scaler.is_enabled()): yp,g,_,pp = model_outputs(model,probe,x)
            with amp_context(device,False):
                total,logs = batch_loss(criterion,yp,yt,g,gt,pp,pt,(step % args.stft_every)==0)
            if not torch.isfinite(total): raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params,args.grad_clip)
            old_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(optimizer); scaler.update()
            stepped = (not scaler.is_enabled()) or scaler.get_scale() >= old_scale
            if stepped:
                for k,v in logs.items():
                    if k == "stft" and (step % args.stft_every) != 0: continue
                    sums[k] = sums.get(k,0.0) + float(v); counts[k] = counts.get(k,0) + 1
        scheduler.step()
        train_metrics = {k:sums[k]/counts[k] for k in sums}
        val_metrics = evaluate(model,probe,val_loader,criterion,device)
        history.append({"epoch":epoch+1,"train":train_metrics,"validation":val_metrics,
                        "lr":scheduler.get_last_lr()[0]})
        if val_metrics["curve"] < best:
            best = val_metrics["curve"]
            save_checkpoint(args.out_dir/"best.pt",model,probe,optimizer,scheduler,scaler,epoch,best,history,args)
        save_checkpoint(args.out_dir/"last.pt",model,probe,optimizer,scheduler,scaler,epoch,best,history,args)
    # Report test metrics for the checkpoint selected on validation, not the final epoch.
    best_checkpoint = safe_torch_load(args.out_dir/"best.pt")
    model.load_state_dict(best_checkpoint["model"], strict=True)
    probe.load_state_dict(best_checkpoint["probe"], strict=True)
    selected_epoch = int(best_checkpoint["epoch"]) + 1
    test_metrics = evaluate(model,probe,test_loader,criterion,device)
    report = {"selected_checkpoint_epoch": selected_epoch,
              "environment":{"python":platform.python_version(),"torch":torch.__version__,
             "numpy":np.__version__,"platform":platform.platform(),"device":str(device),
             "cuda_available":torch.cuda.is_available()},
             "seed":args.seed,"dataset_contract":DATASET_CONTRACT_VERSION,
             "split_ids":{"train":[[args.seed,i] for i in range(args.n_samples)],
                          "validation":[[args.seed+7919,i] for i in range(args.n_val)],
                          "test":[[args.seed+15839,i] for i in range(args.n_test)]},
             "model":{"n_bands":1,"latent_dim":model.stem.out_channels,"frame":FRAME_SIZE},
             "loss_weights":{"stft":W_STFT,"curve":W_CURVE,"param":W_PARAM,"smooth":W_SMOOTH},
             "best_validation_curve":best,"test_metrics":test_metrics,
             "elapsed_seconds":round(time.time()-started,3),"history":history}
    (args.out_dir/"train_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"best_validation_curve":best,"test_metrics":test_metrics},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
