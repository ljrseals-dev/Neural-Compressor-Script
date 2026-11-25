# Missing Module Contracts — Historical Pre-Recovery Audit

**Status:** this document records the pre-recovery interface investigation.
The modules described below have since been recovered and adapted from
`expert-lamp` on branch
`integration/expert-lamp-recovery-2026-11-25`. See
`docs/EXPERT_LAMP_INTEGRATION.md` for the current implementation and verified
CI results. This historical contract is retained to show what the original
training entry point required; its statements that the modules are absent are
not current repository status.

## Ground rules
Contracts are derived from `training/train.py` on this audit branch and the README. Dimensions or algorithms not established by those call sites are marked unknown, not guessed. No stubs or fake tensors are added. Any future temporary stub must be labeled **non-reproducible scaffold** and raise `NotImplementedError` on every functional call.

## Launch-path blocker
The original root-level `training.train.py` did not match the README invocation `python -m training.train`, and its `parents[1]` ROOT expression resolved above the repository. On this audit branch the script has been relocated to `training/train.py`, `training/__init__.py` has been added, and `ROOT = Path(__file__).resolve().parents[1]` now resolves to the repository root. The relocation is source-verified; `training/__init__.py` is present. The launch command has not been executed because required modules are still absent. Note: the script still contains `sys.path.insert(0, str(ROOT))`; this is an explicit path insertion, not an absence of `sys.path` handling. It has not been removed or runtime-tested. The inspected source has the `print(model.count_parameters())` and `if args.strict_semantic:` statements on separate lines; the suspected syntax error from the pasted transition is not present in the repository file.

## 1. model/blocks.py

Imported names:
```python
from model.blocks import ParameterProbe, build_model
```
`NeuralCompressor` is mentioned only in a comment as unused.

### build_model
Observed constructor call: `build_model(n_bands=args.n_bands)` (default 4), then `.to(device)`.

Model is expected to be a torch module and support `.parameters()`, `.train()`, `.eval()`, `.state_dict()`, `.load_state_dict()`, and `.count_parameters()`.

Forward call: `model(x, return_extras=True)`; exactly four returned values:
`y_p, g_p, g_bands, extras`.
- `x`: dry audio; dry-run shape is `[1,1,T]`, where `T=max(256,int(44100*min(duration,0.1))`). Batch input is therefore expected as `[B,1,T]`.
- `y_p`: predicted compressed audio, compared with teacher audio `y_t`.
- `g_p`: predicted gain curve, compared with teacher gain curve `g_t` and passed to smoothness loss.
- `g_bands`: returned but unused by this training script.
- `extras`: mapping containing tensor `"latent"`; its final dimension determines probe input width.
- Outputs must preserve batch dimension. Predictions are explicitly cast to fp32 for loss computation. Dataset targets should be float tensors, preferably fp32.

Unknown and not to be guessed: architecture, depth, hidden size, dropout, activations, receptive field, exact output dimensions, latent rank, gain-band layout, and output bounds. The script passes no hidden-size, depth, or dropout config.

### ParameterProbe
Observed call: `ParameterProbe(latent_dim=latent_dim, n_params=4)`; forward call: `probe(extras["latent"])`.
- Torch module.
- Four outputs correspond in order to `threshold, ratio, attack, release`.
- Return tensor must shape-match teacher `p_t`; likely `[B,4]`, but confirm against actual dataset.
- README describes targets and probe MAE as normalized to `[0,1]`; exact output activation must be checked against target generation before implementation.

### Unit tests required
Random fp32 `x:[B,1,T]`; assert four-item return, tensor types, finite values, batch preservation, latent key, and compatible `y_p/y_t`, `g_p/g_t`, and `p_pred/p_t` shapes. Backpropagate a real scalar loss and assert finite gradients reach both model and probe. Test at least n_bands 1 and 4.

## 2. model/losses.py

Imported names:
```python
from model.losses import CompressorLoss, curve_loss, param_loss, smoothness_loss
```

Observed API:
```python
CompressorLoss(w_stft=1.0, w_curve=0.5, w_param=0.2, w_smooth=0.05)
loss, logs = loss_fn(y_p, y_t, g_p, g_t, p_pred, p_t)
curve_loss(g_p, g_t)
smoothness_loss(g_p)
param_loss(p_pred, p_t)
```

Each standalone loss must return a scalar differentiable tensor. `CompressorLoss` returns `(loss, logs)`; logs must be mapping-like and contain `total, curve, stft, param, smooth`, each supporting `.item()`. The non-STFT path combines `0.5*curve + 0.2*param + 0.05*smooth`; the full path uses all four configured weights.

Objective described in README: imitate a differentiable classical compressor teacher:
- multi-resolution STFT reconstruction between predicted audio and teacher audio;
- gain-curve agreement;
- regression of four teacher compressor parameters;
- gain-curve smoothness regularization.

This is not comparable to the host's C++ heuristic/string demos: they do not run the same audio task or expose the same outputs/metric. No neural-vs-demo comparison is valid yet.

Unknown: STFT resolutions/windows/hops, spectral scaling, reductions, curve reduction, smoothness definition, silence handling, masking (no mask is passed by current caller). Do not invent these.

### Unit tests required
For valid compatible fp32 tensors: finite scalar outputs, intended non-negativity, zero-error behavior, finite prediction gradients, and weighted-log consistency. Toy-fit test should optimize predictions toward a fixed target and verify the documented objective decreases. Include silence, short clip, batch-one, and minimum supported length cases. If a loss is allowed to be negative, document why and revise the proposed non-negative acceptance check explicitly.

## 3. training/dataset.py

Imported names:
```python
from training.dataset import SyntheticCompressorDataset, collate
```

Observed constructor:
```python
SyntheticCompressorDataset(n_samples=args.n_samples, duration_s=args.duration,
                           sample_rate=44100, precompute=True, seed=args.seed)
```

Required dataset behavior:
- Dataset-compatible `__len__` and `__getitem__`.
- Deterministic for fixed configuration and seed.
- `precompute=True` accepted; README says teacher outputs/curves/parameters are precomputed.
- Each sample provides `x, y, g, params`; collate returns a dictionary with those batched keys.
- `x` is dry audio, `y` teacher-compressed audio, `g` teacher gain curve, `params` four teacher parameters in order threshold/ratio/attack/release.
- 44,100 Hz. Default duration 1.5 s. Exact sample rounding and tensor dimensions are not specified.
- Prefer fp32 for all tensors. `x/y` must match audio dimensions; `g` must match model `g_p`; params must match probe output.

Provenance: only synthetic teacher-generated compressor data is stated in README. The actual teacher, parameter ranges/distribution, waveform generator, schema version, and license are not in the available source. These must be documented from the implementation; do not claim a license or no-external-assets status without checking.

Splits: current script creates one dataset, shuffles the training loader, then evaluates on `Subset(ds, list(range(eval_n)))`, where `eval_n = min(args.eval_size, len(ds))`. This is the first N examples of the same dataset used for training; it does not create disjoint train/validation/test sets. Therefore the reported probe MAE is not held-out validation/test performance. Add deterministic, disjoint split IDs and a split seed before treating probe MAE as held-out validation/test performance.

Checkpoint selection: `best.pt` is selected when the epoch's mean training `curve` loss improves (`running["curve"] < best_curve`). It is not selected using held-out validation performance. Until a disjoint validation split exists, “best” means best observed training-curve fit, not best generalization.

The seven public-domain books in the host repo are text corpora, not audio training inputs; they cannot be substituted for this dataset or used to claim a fair audio comparison.

### Unit tests required
Length equals n_samples; sample keys and tensor shapes/dtypes are stable; all values finite; same seed/config yields identical samples; collate preserves order and batches all keys; a batch passes through model and probe; deterministic train/val/test splits have disjoint IDs.

## 4. Additional training-script audit findings

### Resume checkpoint loading
`_try_resume` first calls `torch.load(path, map_location="cpu", weights_only=True)` and retries with `weights_only=False` only for `TypeError` or `RuntimeError`. A weights-only load can fail with an unpickling exception outside that catch tuple (for example, `pickle.UnpicklingError`, depending on the serialized content and PyTorch version). In that case `--resume` can fail before the fallback is reached. Confirm the exact exception against the supported PyTorch versions and handle it deliberately. Only use `weights_only=False` for checkpoints from a trusted source because it enables pickle deserialization.

### Syntax check
The repository version inspected on this branch has the model-parameter `print(...)` and following `if args.strict_semantic:` on separate lines. The syntax-error concern raised from the two-part paste is not confirmed in the tracked file. Still run `python -m py_compile training/train.py` when the environment is available; this has not been run as part of this source audit.

## 5. Acceptance gate
Do not claim acceptance until:
1. [x] Entry-point filename/module path and ROOT corrected on this audit branch: `training/train.py`, `training/__init__.py`, and `parents[1]` resolve consistently. Source inspection confirms the files and path expression; CLI launch/import execution remains untested because functional modules are absent.
2. Restore real implementations (not scaffolds).
3. Pass model shape/gradient tests, finite loss/toy-fit tests, and deterministic dataset/collate tests.
4. Run one epoch on a tiny subset, e.g. `--epochs 1 --n-samples 8 --batch-size 2 --eval-size 4 --duration 0.05 --no-amp --num-workers 0 --stft-every 1`, after confirming minimum audio length support.
5. Repeat from clean output directories with same seed; compare data fingerprints, initial weights, history, and final weights. Document deterministic-algorithm settings and tolerances.
6. Record teacher/version, dataset schema/provenance/license, commit, Python/PyTorch/NumPy versions, device, and seed.
7. Do not compare with C++ heuristic demos until same task, inputs, and defensible common metric exist.

## Status
- [x] Enumerated imported names.
- [x] Recorded signatures and return structure visible at call sites.
- [x] Recorded observable shapes/roles and unknown dimensions.
- [x] Identified and corrected launch-path mismatch on this branch; runtime launch remains untested.
- [ ] Restore or author real model, loss, and teacher/dataset implementations.
- [ ] Define dataset provenance/license and deterministic splits.
- [ ] Add and execute tests.
- [ ] Complete two reproducible one-epoch runs.

**Conclusion:** this is a contract, not an implementation. No stubs, fabricated data, training run, trained model, or benchmark result is claimed.
