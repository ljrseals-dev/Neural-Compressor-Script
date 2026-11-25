# Recovered expert-lamp integration

Status: implementation authored on `integration/expert-lamp-recovery-2026-11-25`.
Runtime tests and training have **not** yet been executed.

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
   computed separately after training.
6. Checkpoints are versioned and loaded only with `weights_only=True`.
   There is no unrestricted pickle fallback. Legacy checkpoints require an
   explicit trusted offline migration; they are not silently loaded.

## Semantic hook

`--strict-semantic` now fails with an explicit explanation because
`harmonia_neural.semantic_postcheck` was not found in the inspected
HarmoniaCodec-Music source. Harmonia semantic components are not silently
substituted into the compressor.

## Verification still required

- Run `python -m py_compile model/blocks.py model/losses.py training/dataset.py training/train.py`.
- Run `pytest -q tests/test_recovered_integration.py`.
- Run a CPU smoke: `python -m training.train --epochs 1 --n-samples 8 --n-val 4 --n-test 4 --batch-size 2 --duration 0.25 --no-amp --device cpu --out-dir /tmp/nc-smoke`.
- Repeat with the same seed into a clean output directory and compare split IDs,
  histories, and state dictionaries.
- Record actual Python/PyTorch versions and runtime outcomes before acceptance.

This document records authored source, not successful execution, convergence,
perceptual quality, or a neural-versus-classical performance result.
