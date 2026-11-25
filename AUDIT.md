# Neural Compressor Audit — Updated Disposition

Date: 2026-11-25
Repository: `ljrseals-dev/Neural-Compressor-Script`

**Status:** recovered implementation authored, CI-verified, and awaiting review
in [draft PR #1](https://github.com/ljrseals-dev/Neural-Compressor-Script/pull/1).
The `main` branch has not been modified or merged.

## Owner decision now recorded

The owner directed that the functional modules be taken from
`ljrseals-dev/expert-lamp` and integrated into
`ljrseals-dev/Neural-Compressor-Script`.

The selected teacher is expert-lamp's classical single-band RMS/soft-knee
compressor. It remains an offline target generator, not a differentiable
teacher-training graph. The implementation is adapted to the original
four-value forward contract; it is not represented as a byte-for-byte copy.

## Recovered modules

- `model/blocks.py`: dilated CNN, latent output, and `ParameterProbe`.
- `model/losses.py`: multi-resolution STFT, curve, parameter, and smoothness
  losses with the original Neural-Compressor-Script weights.
- `training/dataset.py`: deterministic synthetic audio, classical teacher,
  precompute cache, frame-aligned gain targets, and collate.
- `training/train.py`: compatible four-value model call, explicit single-band
  constraint, disjoint train/validation/test sample IDs, validation-selected
  checkpoint, separate test evaluation, and strict versioned resume loading.

## Corrections applied

- Teacher and student gain curves now share the same 256-sample frame windows.
  The loss rejects shape mismatches; it does not truncate unrelated time grids.
- RMS envelope padding preserves input length for the even 512-sample window.
- The model rejects `n_bands != 1` because the recovered teacher has only one
  gain target. No unsupported multi-band head is invented.
- Resume loading requires `weights_only=True`; unrestricted
  `weights_only=False` fallback is removed.
- Test metrics are computed from the checkpoint selected on validation data.
- `semantic_postcheck` remains unsupported and `--strict-semantic` fails
  explicitly rather than silently assuming the missing Harmonia function.

## Verification evidence

GitHub Actions on `ec3daefb27f4b8d02d3a00731bd77a8db1eaf22e` passed:

- Python 3.11 syntax compilation.
- 6 integration tests, including model gradient flow, multiple sample rates,
  precompute equivalence, frame alignment, mismatch rejection, and toy-fit loss
  reduction.
- One-epoch CPU smoke with 8 train, 4 validation, and 4 test samples.
- A second identical run with exact matching split IDs, histories,
  model/probe weights, optimizer/scheduler/scaler states, validation metric,
  and test metrics.

This verifies a small functional path and repeatability on the CI runner. It
does **not** establish convergence, generalization, perceptual quality, or
superiority to the classical teacher.

## Repository boundaries

`Neural-Compressor-Script` remains the source of truth for the neural
compressor. `Compression-Progress-AI` remains a separate C++ heuristic/text
demonstration repository and should consume only a later evaluation artifact
when task, inputs, corpus, and metric are shared. `HarmoniaCodec-Music` remains
separate; its semantic components are not part of this compressor contract.

## Outstanding owner/review items

- Review and decide whether to merge draft PR #1.
- Resolve the README All Rights Reserved notice versus full MIT LICENSE policy.
  No license file was modified in this integration.
- Decide whether future work requires a real multi-band teacher.
- Define a rights-cleared real-audio evaluation set and quality metrics before
  making any performance claims.
