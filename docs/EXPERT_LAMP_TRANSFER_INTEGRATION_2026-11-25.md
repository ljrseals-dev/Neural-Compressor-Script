# expert-lamp module transfer and integration record
Date: 2026-11-25
Branch: integration/expert-lamp-module-transfer-2026-11-25
Status: integration authored; **not runtime-tested**

## Provenance
Adapted from owner-controlled repository `ljrseals-dev/expert-lamp`, branch `main`:
- `model/blocks.py`, blob `8f74db9fbc77f3b77c70781582d94441b472ac76`
- `model/losses.py`, blob `1da3dcad990d8b2e0d2821209832ac153b8a3ad2`
- `training/dataset.py`, blob `70587afd5ce46de7b3963398c3638e63cce10405`

The modules are adapted rather than copied byte-for-byte to satisfy Neural-Compressor-Script's legacy call sites. The repository owner directed transfer and integration.

## Adaptation
- `model/blocks.py`: student CNN and gain curve; latent pooled features; `ParameterProbe`; four-item forward contract; `build_model`; single-band only.
- `model/losses.py`: expert-lamp spectral/gain/parameter/smoothness concepts exposed via the existing six-argument training call and `(loss, logs)` return.
- `training/dataset.py`: deterministic synthetic examples, `precompute`, `collate`, and teacher targets. Teacher curve is upsampled to full rate then pooled using 256-sample frames to align with the student's frame grid.
- `training/train.py`: `--n-bands` default changed to 1; unrestricted pickle fallback removed from resume loading; semantic post-check skips if the absent `semantic_postcheck` function is not callable.
- README adjusted for single-band contract and optional semantic check.

## Known remaining risks/blockers
1. **Syntax/runtime verification pending.** None of the files have been executed or tests run in this environment. Integration is source-authored only.
2. **Validation/test protocol remains incomplete.** Training and validation use explicit non-overlapping index ranges from a single deterministic synthetic dataset, and best checkpoint selection now uses validation gain-curve loss. There is no separate test split yet, no formal persisted split manifest, and validation is still limited to the generated synthetic distribution.
3. **Checkpoint contract.** Safe `weights_only=True` resume may reject legacy checkpoint payloads. No unsafe fallback is retained; legacy conversion must be offline and explicit.
4. **Teacher alignment is improved but needs tests.** Teacher curve is mapped through the full-rate interpolation and pooled on the same 256-sample frame size. Tests must verify timestamp conventions, tail padding, varying sample rates/durations, and gain target shapes.
5. **Single-band limitation.** `n_bands != 1` is rejected. No multi-band targets or claims are introduced.
6. **Parameter-head gradient path.** Probe takes pooled CNN latent. Confirm all intended model/probe parameters receive finite gradients in tests.
7. **STFT minimum-length and boundary behavior.** Must test short clips, silence, and batch-one.
8. **Data provenance.** Teacher and synthetic waveform generation are now inherited/adapted from expert-lamp; document version and sample ID split contract before quality claims.

## Next integration gates
- Source-review all files for syntax/API consistency.
- Add tests for shapes, deterministic samples, collate, aligned gain targets, finite gradients, and toy-fit decrease. The initial transfer tests cover deterministic sample equivalence, model/probe shapes, finite gradient presence, and the single-band rejection; these tests have not been executed.
- Add a held-out test split and persist split IDs/fingerprints; validate data integrity and confirm checkpoint selection uses the intended validation metric.
- Run two clean CPU smoke/reproducibility runs with `--n-bands 1`, `--no-amp`, and short duration; then report logs without implying convergence.
- Review and PR. Do not merge or benchmark without the owner's later direction.

## License policy
Owner confirmation authorizes their code transfer. This record does not modify license files or resolve the README/LICENSE notice consistency issue.
