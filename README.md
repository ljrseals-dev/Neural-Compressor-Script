Copyright © 2026 Lewis James Seals Jr. All rights reserved.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.

README — Neural Compressor Training


![Neural Compressor Artwork](/images/neural_compressor.jpg)

Training harness for the neural compressor. A differentiable classical compressor serves as the teacher target; the neural model is trained end-to-end to reproduce the teacher's gain curve and static parameters (threshold, ratio, attack, release).

---

1. Overview

The script trains two modules jointly:

Component Role
model.blocks.build_model(n_bands) Produces gain curves g_p, gain bands g_bands, and a latent embedding from the input audio.
model.blocks.ParameterProbe(latent_dim, 4) Regresses the four compressor parameters from the latent.

Supervision comes from training.dataset.SyntheticCompressorDataset, which precomputes:

· x   — input audio (dry)
· y   — classical compressed reference
· g   — classical gain curve
· params — static compressor parameters

Losses are combined with fixed weights defined at module top (W_STFT, W_CURVE, W_PARAM, W_SMOOTH). The STFT loss is the expensive one, so it is only computed every --stft-every steps; non-STFT steps use the cheap curve + param + smooth combination.

---

2. Requirements

· Python ≥ 3.10
· torch (CUDA build recommended; CPU works but is slow)
· numpy
· Optionally harmonia_neural (only for --strict-semantic)

The project must be importable from the repo root — the script prepends ROOT (the parent of training/) to sys.path, so running from the repo root is the supported entry point.

---

3. Quick start

From the repository root:

```bash
# Basic training run (20 epochs, 256 synthetic samples)
python -m training.train

# Longer run, bigger batch
python -m training.train --epochs 30 --batch-size 8 --n-samples 512

# Resume from checkpoints/last.pt
python -m training.train --resume

# Disable mixed precision (useful on CPU or for debugging NaNs)
python -m training.train --no-amp

# Enforce the semantic backend post-check
python -m training.train --strict-semantic
```

The device is auto-selected (cuda if available, otherwise cpu). Override with --device.

---

4. CLI reference

Flag Default Description
--epochs 20 Number of epochs. Must be ≥ 1.
--batch-size 8 Batch size for the training loader.
--n-samples 256 Number of synthetic samples to generate.
--duration 1.5 Duration of each generated clip (seconds).
--lr 2e-4 Peak learning rate (AdamW, weight decay 1e-4).
--n-bands 4 Number of gain bands produced by the model.
--latent-dim inferred Width of extras["latent"]. Inferred from the model (via attributes, or a dry-run forward pass) if omitted.
--device auto cuda or cpu.
--out-dir checkpoints/ Where to write last.pt, best.pt, train_report.json.
--seed 42 Seeds random, numpy, torch, CUDA, and the DataLoader generator.
--stft-every 4 Compute STFT loss every N optimizer steps. Clamped to ≥ 1.
--grad-clip 5.0 Gradient norm clip. 0 disables clipping.
--num-workers 0 DataLoader workers for the training loader.
--eval-size 64 Number of samples used for the per-epoch probe-MAE report (capped by --n-samples).
--resume off Resume from <out-dir>/last.pt. Restores model, probe, optimizer, scheduler, epoch, best_curve, and history.
--no-amp off Disables AMP. AMP only actually activates on CUDA.
--strict-semantic off Calls harmonia_neural.semantic_postcheck(model) after model build, if the package is importable. Silently skipped otherwise.

---

5. Outputs

Written to --out-dir:

File When Contents
last.pt Every epoch, and on KeyboardInterrupt model, probe, optim, sched, epoch, best_curve, history, args
best.pt Whenever the epoch's mean curve loss improves model, probe, args, probe_mae, epoch
train_report.json End of training final_probe_mae, best_curve_loss, param_count, full history

history is a list of per-epoch dicts with:

```json
{
  "epoch": 1,
  "loss": {"total": ..., "curve": ..., "stft": ..., "param": ..., "smooth": ...},
  "probe_mae": {"threshold": ..., "ratio": ..., "attack": ..., "release": ..., "mean": ...},
  "lr": ...,
  "time_s": ...
}
```

Note on averaging: loss.stft is averaged over STFT-computed steps only, so its magnitude is comparable to curve and param — not artificially deflated by non-STFT steps. All other loss fields (and total) are averaged over all batches.

---

6. Loss weights

Defined as module-level constants at the top of training/train.py and used by both loss branches (full CompressorLoss on STFT steps, and the reduced curve + param + smooth on non-STFT steps):

Constant Weight Loss
W_STFT 1.0 Multi-resolution STFT reconstruction
W_CURVE 0.5 Gain-curve MSE (per band)
W_PARAM 0.2 Probe-vs-teacher parameter MSE
W_SMOOTH 0.05 Smoothness penalty on the gain curve

To tune, edit the constants at the top of the script — both branches stay in sync.

---

7. Training loop at a glance

1. Seed everything (random, numpy, torch, CUDA, DataLoader generator).
2. Build model with build_model(n_bands) and infer latent_dim.
3. Build ParameterProbe(latent_dim=latent_dim, n_params=4).
4. Generate SyntheticCompressorDataset with precompute=True.
5. Build a shuffled training loader (drop_last=True) and a fixed, unshuffled eval subset of --eval-size samples.
6. Optimizer: AdamW over model.parameters() + probe.parameters() (probe is included in the grad-clip group).
7. Scheduler: CosineAnnealingLR(T_max=epochs).
8. Per step:
   · Forward under autocast.
   · Every --stft-every-th step: full CompressorLoss.
   · Otherwise: W_CURVE*curve + W_PARAM*param + W_SMOOTH*smooth.
   · Backward with GradScaler, clip, step, update.
9. Per epoch:
   · Average running losses.
   · Compute probe MAE over the fixed eval subset.
   · Log a row to history.
   · Save best.pt if curve improved; always save last.pt.

---

8. Resume behavior

--resume reads <out-dir>/last.pt if present:

· model, probe: loaded strictly.
· optim, sched: loaded leniently — a shape mismatch or missing key triggers a warning and fresh optimizer/scheduler state rather than a crash.
· epoch, best_curve, history: restored, and training continues from epoch + 1.
· If start_epoch > --epochs, the script exits without training.

If last.pt is absent, a fresh run starts and a notice is printed.

---

9. Sanity checks & warnings

The script fails fast or warns for the following conditions:

· --epochs < 1, --batch-size < 1, --n-samples < 1 → SystemExit.
· --stft-every < 1 → clamped to 1 with a warning.
· --batch-size > --n-samples → warning.
· --grad-clip < 0 → warning; clipping disabled.
· Empty training DataLoader (e.g. batch_size > n_samples under drop_last=True) → SystemExit with suggested fix.

---

10. Interruption & recovery

· Ctrl-C (SIGINT) saves last.pt before re-raising, so a subsequent --resume picks up mid-run.
· Other signals (SIGTERM, OOM kill) do not save automatically — use --resume from the last completed epoch.

---

11. AMP details

· AMP only activates when --device resolves to CUDA and --no-amp is not passed.
· The forward pass runs inside torch.autocast.
· Both STFT loss and the cheap loss path run under torch.autocast(..., enabled=False), forcing fp32 — STFT in fp16 is numerically fragile.
· torch.amp.GradScaler(device.type, ...) is used when available, falling back to torch.cuda.amp.GradScaler on older PyTorch.

---

12. Semantic backend post-check

With --strict-semantic, the script calls harmonia_neural.semantic_postcheck(model) right after the model is built. Skipped (with a notice) if:

· harmonia_neural cannot be imported, or
· the model has no .sem attribute.

---

13. Troubleshooting

Symptom Likely cause / fix
DataLoader is empty Lower --batch-size or raise --n-samples.
probe_mae.mean is nan Empty eval loader — shouldn't happen; check --eval-size.
loss.stft much larger than curve Normal early on; the loss weight balance is intentional.
Loss goes to NaN Try --no-amp, lower --lr, or raise --grad-clip.
Training is very slow Ensure --device cuda; AMP is on by default. Also raise --num-workers and --batch-size if VRAM allows.
--resume doesn't pick up Confirm <out-dir>/last.pt exists; check the printed [resume] line.

---

14. Reproducibility

Given the same --seed, --n-samples, --n-bands, and code revision, the dataset and loader ordering are deterministic. Non-determinism can still enter from:

· cuDNN autotuning (not disabled here), and
· torch.use_deterministic_algorithms (not enabled here).

If bit-exact reproducibility is required, set torch.backends.cudnn.deterministic = True and torch.use_deterministic_algorithms(True) at the top of main().
