# Neural Compressor Audit
Date: 2026-11-25
Repository: `ljrseals-dev/Neural-Compressor-Script`
Status: **architecture and teacher decision pending; no training acceptance claim**

## Repository responsibility

This repository is the source of truth for the neural compressor: teacher definition, deterministic audio dataset, model, losses, training entry point, checkpoints, and neural-specific tests.

`ljrseals-dev/Compression-Progress-AI` is a separate repository containing heuristic/text-oriented C++ demonstration programs and their audit/benchmark material. It should **consume a later, reproducible evaluation artifact or adapter** from this repository only after both systems share a defined task, input corpus, and defensible metric. It should not host the neural model implementation or training pipeline, and the current demos are not a valid neural-compressor benchmark.

## Teacher decision: unresolved

The training script and README describe a differentiable classical compressor teacher, but the actual teacher implementation and its precise behavior have not been recovered or verified in the inspected repository content.

No choice is made yet between:
1. **Recover the original teacher** — preferred if its source, provenance, parameterization, and intended behavior can be located and verified; or
2. **Design a new teacher** — only with explicit approval and a separately versioned specification describing transfer function, detector, attack/release behavior, parameter ranges, gain-curve calculation, and reference tests.

Do not implement `SyntheticCompressorDataset`, `CompressorLoss`, or claim teacher fidelity until this decision is recorded and the teacher behavior is testable. Do not silently substitute a conventional compressor and call it the original.

## Entry point repair

The prior root-level `training.train.py` filename did not match the documented `python -m training.train` invocation. It also calculated ROOT as the parent of the repository.

The audit branch now places the entry point at `training/train.py`, adds `training/__init__.py`, and removes the root-level file. From that location, `Path(__file__).resolve().parents[1]` resolves to the repository root, so the existing module invocation and ROOT calculation are aligned.

This is a source-layout correction only. It does not make training runnable: `model.blocks`, `model.losses`, and `training.dataset` still require real implementations.

## Implementation order and gates

1. Recover and verify the original teacher, or approve a new teacher specification.
2. Implement deterministic `training/dataset.py` using the verified teacher; record sample schema, seed behavior, provenance, license, and disjoint train/validation/test splits.
3. Implement `model/losses.py` against the teacher's waveform/gain/parameter targets; test finite values, gradients, and toy-fit decrease.
4. Implement `model/blocks.py` against the agreed tensor shapes and task; test forward shapes and gradient flow.
5. Run a tiny one-epoch acceptance test from a clean checkout, then repeat with the same seed and compare fingerprints, history, and model state under documented deterministic settings.
6. Only after those gates, publish a benchmark artifact for possible consumption by `Compression-Progress-AI`. No cross-repository score before same-task/same-data/same-metric validation.

## Current verified status

- Entry point relocated to the documented package path on the audit branch.
- Module initializer added.
- Missing-module contract is documented separately in `docs/MISSING_MODULE_CONTRACT_2026-11-25.md`.
- Teacher recovery/design decision remains open.
- The three functional modules remain absent; no scaffold or fake implementation is accepted.
- No end-to-end training, reproducibility result, trained checkpoint, or neural-vs-heuristic comparison is claimed.
