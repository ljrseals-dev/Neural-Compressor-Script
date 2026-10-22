#!/usr/bin/env python3
"""
Training script for the neural compressor.

Uses the differentiable classical compressor as the target (route 1).
Precomputes gain curves; STFT loss is evaluated every N steps on a subset
to keep wall-clock reasonable on a single GPU / CPU.

Usage:
  python -m training.train --epochs 30 --batch-size 8 --n-samples 256
  python -m training.train --resume              # continue from last.pt
  python -m training.train --no-amp              # disable mixed precision
  python -m training.train --strict-semantic     # enforce semantic backend

Revisions vs the previous version:
  * Probe is included in grad clipping (previously clipped only model params).
  * curve/smooth/param losses imported at module level, not inside the step loop.
  * Loss weights factored to module constants shared by both loss branches.
  * ParameterProbe latent width inferred from a dry-run forward pass; --latent-dim
    overrides it. n_bands no longer assumed to equal latent width.
  * running["stft"] averaged over STFT-computed steps only (was divided by all
    batches and therefore reported ~stft_every too small).
  * --stft-every 0 clamped to 1 (was ZeroDivisionError).
  * Full seeding (random, numpy, torch, cuda) and a seeded DataLoader generator.
  * last.pt written every epoch; --resume restores optimizer, scheduler, epoch.
  * KeyboardInterrupt saves last.pt before re-raising.
  * Sanity checks for: empty loader, epochs<1, batch_size>n_samples + drop_last.
  * Separate eval loader (shuffle=False, drop_last=False, capped subset).
  * Optional AMP (CUDA only) with STFT loss computed in fp32.
  * Optional --strict-semantic post-check if harmonia_neural is importable.
  * "--no-amp" flag (amp defaults to on).
  * "--grad-clip" flag (default 5.0).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# allow running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.blocks import ParameterProbe, build_model          # NeuralCompressor unused
from model.losses import (
    CompressorLoss,
    curve_loss,
    param_loss,
    smoothness_loss,
)
from training.dataset import SyntheticCompressorDataset, collate


# ------------------------------------------------------------------ constants

# Single source of truth for loss weights. Both the full CompressorLoss path
# and the cheap curve+param+smooth path use these numbers; do not hardcode
# the weights a second time.
W_STFT = 1.0
W_CURVE = 0.5
W_PARAM = 0.2
W_SMOOTH = 0.05

PROBE_PARAM_NAMES = ("threshold", "ratio", "attack", "release")


# ------------------------------------------------------------------ args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-bands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=None,
                   help="width of extras['latent']; inferred if omitted")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stft-every", type=int, default=4,
                   help="compute STFT loss every N steps (clamped to >=1)")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-size", type=int, default=64,
                   help="number of samples used for the per-epoch probe-MAE report")
    p.add_argument("--resume", action="store_true",
                   help="resume from <out-dir>/last.pt")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="disable CUDA mixed precision")
    p.set_defaults(amp=True)
    p.add_argument("--strict-semantic", action="store_true",
                   help="run harmonia_neural.semantic_postcheck on the built model")
    return p.parse_args()


# ------------------------------------------------------------------ helpers

def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _sanity_check_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.n_samples < 1:
        raise SystemExit("--n-samples must be >= 1")
    if args.stft_every < 1:
        print(f"  [warn] --stft-every={args.stft_every} clamped to 1")
        args.stft_every = 1
    if args.eval_size < 1:
        args.eval_size = 1
    if args.batch_size > args.n_samples:
        print(f"  [warn] batch_size ({args.batch_size}) > n_samples ({args.n_samples})")
    if args.grad_clip < 0:
        print(f"  [warn] --grad-clip={args.grad_clip} <= 0 disables clipping")
        args.grad_clip = 0.0


def _infer_latent_dim(model: nn.Module, args: argparse.Namespace,
                      device: torch.device) -> int:
    """Determine the width of extras['latent'] without trusting n_bands."""
    if args.latent_dim is not None and args.latent_dim > 0:
        return int(args.latent_dim)
    for attr in ("latent_dim", "embed_dim", "d_model"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    # Dry-run with a short dummy input.
    T = max(256, int(44100 * min(args.duration, 0.1)))
    dummy = torch.zeros(1, 1, T, device=device)
    model.eval()
    with torch.no_grad():
        _, _, _, extras = model(dummy, return_extras=True)
    latent_dim = int(extras["latent"].shape[-1])
    model.train()
    return latent_dim


def _try_resume(path: Path, model: nn.Module, probe: nn.Module,
                optim: torch.optim.Optimizer, sched) -> tuple[int, float, list]:
    """Load <path> if present; return (next_epoch, best_curve, history)."""
    if not path.exists():
        print(f"  [resume] no checkpoint at {path}; starting fresh")
        return 1, float("inf"), []
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    probe.load_state_dict(ckpt["probe"])
    if "optim" in ckpt:
        try:
            optim.load_state_dict(ckpt["optim"])
        except (ValueError, KeyError) as e:
            print(f"  [resume] could not restore optimizer ({e}); fresh optimizer state")
    if "sched" in ckpt:
        try:
            sched.load_state_dict(ckpt["sched"])
        except (ValueError, KeyError) as e:
            print(f"  [resume] could not restore scheduler ({e}); fresh scheduler state")
    best = float(ckpt.get("best_curve", float("inf")))
    history = list(ckpt.get("history", []))
    next_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"  [resume] loaded {path} (epoch {next_epoch - 1}, best_curve={best:.4f})")
    return next_epoch, best, history


def _save_checkpoint(path: Path, model: nn.Module, probe: nn.Module,
                     optim, sched, epoch: int, best_curve: float,
                     history: list, args: argparse.Namespace) -> None:
    torch.save({
        "model": model.state_dict(),
        "probe": probe.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "best_curve": best_curve,
        "history": history,
        "args": vars(args),
    }, path)


def _strict_semantic_check(model: nn.Module) -> None:
    try:
        import harmonia_neural as hn        # type: ignore
    except ImportError:
        print("  [strict-semantic] harmonia_neural not importable; skipping")
        return
    if not hasattr(model, "sem"):
        print("  [strict-semantic] model has no .sem attribute; skipping")
        return
    hn.semantic_postcheck(model)
    print("  [strict-semantic] OK")


# ------------------------------------------------------------------ main

def main() -> None:
    args = parse_args()
    _sanity_check_args(args)

    device = torch.device(args.device)
    _seed_all(args.seed, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(args.amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        # Older torch: no torch.amp.GradScaler(device, ...) signature.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Device: {device}   amp={use_amp}   seed={args.seed}")
    print(f"Building model (n_bands={args.n_bands}) ...")
    model = build_model(n_bands=args.n_bands).to(device)
    latent_dim = _infer_latent_dim(model, args, device)
    print(f"  inferred latent_dim={latent_dim}")
    probe = ParameterProbe(latent_dim=latent_dim, n_params=4).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    if args.strict_semantic:
        _strict_semantic_check(model)

    loss_fn = CompressorLoss(
        w_stft=W_STFT, w_curve=W_CURVE, w_param=W_PARAM, w_smooth=W_SMOOTH,
    )

    print("Generating synthetic dataset ...")
    ds = SyntheticCompressorDataset(
        n_samples=args.n_samples,
        duration_s=args.duration,
        sample_rate=44100,
        precompute=True,
        seed=args.seed,
    )

    # Seeded generator so shuffle=True is reproducible across runs.
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=True,
        generator=g,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"DataLoader is empty (n_samples={args.n_samples}, "
            f"batch_size={args.batch_size}, drop_last=True). "
            "Reduce --batch-size or increase --n-samples."
        )

    # Fixed, unshuffled eval subset for the per-epoch probe-MAE report.
    eval_n = min(args.eval_size, len(ds))
    eval_loader = DataLoader(
        Subset(ds, list(range(eval_n))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )

    all_params = list(model.parameters()) + list(probe.parameters())
    optim = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    start_epoch = 1
    best_curve = float("inf")
    history: list = []
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    if args.resume:
        start_epoch, best_curve, history = _try_resume(
            last_path, model, probe, optim, sched
        )
    if start_epoch > args.epochs:
        print(f"start_epoch ({start_epoch}) > --epochs ({args.epochs}); nothing to do")
        return

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            probe.train()
            t0 = time.time()
            running = {"total": 0.0, "curve": 0.0, "stft": 0.0,
                       "param": 0.0, "smooth": 0.0}
            n_batches = 0
            n_stft_steps = 0

            for step, batch in enumerate(loader):
                x = batch["x"].to(device, non_blocking=True)
                y_t = batch["y"].to(device, non_blocking=True)
                g_t = batch["g"].to(device, non_blocking=True)
                p_t = batch["params"].to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)

                with torch.autocast(device.type, enabled=use_amp):
                    y_p, g_p, g_bands, extras = model(x, return_extras=True)
                    p_pred = probe(extras["latent"])

                # STFT runs in fp32 even under autocast.
                with torch.autocast(device.type, enabled=False):
                    if step % args.stft_every == 0:
                        loss, logs = loss_fn(
                            y_p.float(), y_t, g_p.float(), g_t,
                            p_pred.float(), p_t,
                        )
                        n_stft_steps += 1
                    else:
                        l_curve = curve_loss(g_p.float(), g_t)
                        l_smooth = smoothness_loss(g_p.float())
                        l_param = param_loss(p_pred.float(), p_t)
                        loss = (
                            W_CURVE * l_curve
                            + W_PARAM * l_param
                            + W_SMOOTH * l_smooth
                        )
                        logs = {
                            "total": loss.item(),
                            "curve": l_curve.item(),
                            "stft": 0.0,
                            "param": l_param.item(),
                            "smooth": l_smooth.item(),
                        }

                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                scaler.step(optim)
                scaler.update()

                for k in running:
                    running[k] += logs.get(k, 0.0)
                n_batches += 1

            sched.step()

            # STFT is averaged over STFT-computed steps only.
            for k in running:
                if k == "stft":
                    running[k] /= max(n_stft_steps, 1)
                else:
                    running[k] /= max(n_batches, 1)

            # Per-epoch probe-MAE on a fixed, unshuffled subset.
            model.eval()
            probe.eval()
            all_pred, all_tgt = [], []
            with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                for batch in eval_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    p_t = batch["params"].to(device, non_blocking=True)
                    _, _, _, extras = model(x, return_extras=True)
                    p_pred = probe(extras["latent"])
                    all_pred.append(p_pred.float().cpu())
                    all_tgt.append(p_t.float().cpu())
            if all_pred:
                pred = torch.cat(all_pred, 0)
                tgt = torch.cat(all_tgt, 0)
                mae = (pred - tgt).abs().mean(dim=0)
                probe_mae = {name: float(mae[i])
                             for i, name in enumerate(PROBE_PARAM_NAMES)}
                probe_mae["mean"] = float(mae.mean())
            else:
                probe_mae = {name: float("nan") for name in PROBE_PARAM_NAMES}
                probe_mae["mean"] = float("nan")

            elapsed = time.time() - t0
            row = {
                "epoch": epoch,
                "loss": running,
                "probe_mae": probe_mae,
                "lr": sched.get_last_lr()[0],
                "time_s": elapsed,
            }
            history.append(row)
            print(
                f"Epoch {epoch:3d} | total {running['total']:.4f} "
                f"curve {running['curve']:.4f} stft {running['stft']:.4f} "
                f"param {running['param']:.4f} | probe MAE {probe_mae['mean']:.3f} "
                f"| {elapsed:.1f}s"
            )

            if running["curve"] < best_curve:
                best_curve = running["curve"]
                torch.save({
                    "model": model.state_dict(),
                    "probe": probe.state_dict(),
                    "args": vars(args),
                    "probe_mae": probe_mae,
                    "epoch": epoch,
                }, best_path)
                print(f"  -> saved best.pt (curve={best_curve:.4f})")

            _save_checkpoint(
                last_path, model, probe, optim, sched,
                epoch, best_curve, history, args,
            )
    except KeyboardInterrupt:
        print("\nInterrupted; saving last.pt ...")
        _save_checkpoint(
            last_path, model, probe, optim, sched,
            epoch if "epoch" in locals() else start_epoch,
            best_curve, history, args,
        )
        raise

    report = {
        "final_probe_mae": history[-1]["probe_mae"],
        "best_curve_loss": best_curve,
        "param_count": model.count_parameters(),
        "history": history,
    }
    with open(out_dir / "train_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== Probe MAE (normalized [0,1] params) ===")
    for k, v in history[-1]["probe_mae"].items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\nCheckpoints -> {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
