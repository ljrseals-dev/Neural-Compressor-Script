# Recovered expert-lamp integration

**Status:** source integrated and CI-verified on
`integration/expert-lamp-recovery-2026-11-25`.
**Review:** draft PR #1, not merged.

## Source and target

The single-band model, teacher, synthetic signal generator, and loss foundations
were recovered from `ljrseals-dev/expert-lamp` and adapted to the original
`Neural-Compressor-Script` training entry point. The working contract is:

- `build_model(n_bands=1)` returns a mono, single-band CNN.
- `model(x, return_extras=True)` returns
  `(y_pred, g_pred, g_bands, extras)`.
- `extras["latent"]` feeds `ParameterProbe(latent_dim, n_params=4)`.
- The dataset returns dry audio, teacher audio, teacher gain frames, normalized
  threshold/ratio/attack/release parameters, and deterministic sample IDs.
- `CompressorLoss(y_pred, y_tgt, g_pred, g_tgt, p_pred, p_tgt)` returns
  `(loss, logs)`.

## Important adaptations

1. The teacher remains an offline classical target generator; it is not claimed
   to be differentiable.
2. The teacher gain is interpolated to full sample rate and then averaged over
   the exact 256-sample windows used by the student. The final partial window
   uses ceil-mode pooling on both sides. The loss rejects mismatched curve
   shapes instead of truncating them.
3. The current teacher is single-band. `--n-bands` defaults to 1 and values
   other than 1 fail explicitly; no synthetic multi-band target is fabricated.
4. The original Neural-Compressor-Script loss weights are retained:
   STFT=1.0, curve=0.5, parameter=0.2, smoothness=0.05.
5. Train, validation, and test use distinct deterministic seed/index IDs.
   Best-checkpoint selection uses validation curve loss; test metrics are
   computed separately using the selected best checkpoint.
6. Checkpoints are versioned and loaded only with `weights_only=True`.
   There is no unrestricted pickle fallback. Legacy checkpoints require an
   explicit trusted offline migration; they are not silently loaded.
7. RMS envelope padding is asymmetric for even window lengths so the envelope
   has exactly the same sample length as the input waveform.

## Semantic hook

`--strict-semantic` now fails with an explicit explanation because
`harmonia_neural.semantic_postcheck` was not found in the inspected
HarmoniaCodec-Music source. Harmonia semantic components are not silently
substituted into the compressor.

## Verification completed

GitHub Actions on head commit
`ec3daefb27f4b8d02d3a00731bd77a8db1eaf22e` completed successfully:

- `python -m py_compile model/blocks.py model/losses.py training/dataset.py training/train.py`: passed.
- `pytest -q tests/test_recovered_integration.py`: **6 passed**.
- CPU one-epoch training smoke: passed.
- Second identical CPU smoke: passed.
- Reproducibility comparison: split IDs, histories, model/probe states,
  optimizer/scheduler/scaler states, selected validation metric, and test
  metrics matched exactly.

Smoke configuration: 8 training samples, 4 validation samples, 4 test samples,
0.25-second clips, batch size 2, CPU, no AMP, deterministic mode.

The run produced a finite validation/test report and a usable checkpoint. These
numbers are smoke evidence only; they are not a convergence, perceptual-quality,
or neural-versus-classical performance benchmark.

## Remaining work

- Run longer training with larger synthetic datasets and inspect learning
  curves for instability or underfitting.
- Evaluate on rights-cleared real audio and define perceptual/technical metrics.
- Establish whether multi-band supervision is needed; the current model
  deliberately supports one band only.
- Resolve the README All Rights Reserved notice versus the full MIT LICENSE
  policy under the repository owner's direction. No license files were changed.
- Keep the PR draft until the owner reviews the implementation and decides
  whether to merge it.
