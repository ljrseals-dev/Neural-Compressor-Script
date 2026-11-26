# Training Evaluation Audit Addendum
Date: 2026-11-25
Repository: ljrseals-dev/Neural-Compressor-Script
Branch: audit/missing-module-contract-2026-11-25

## Direct source inspection

Inspected `training/train.py` at this branch (blob `7b5b4667dd6342acf1689a5b5b6a88f1001403a1`). The per-epoch evaluation loader is constructed with:

```python
eval_n = min(args.eval_size, len(ds))
eval_loader = DataLoader(
    Subset(ds, list(range(eval_n))),
    ...
)
```

The training loader is built from that same `ds`, with `shuffle=True`. Disabling shuffle in the evaluation loader does not make the examples disjoint. Consequently, the printed and saved probe MAE is measured on examples also available to training. It is training-subset MAE, **not held-out validation or test performance**. Do not use it as evidence of generalization.

Required correction before reporting model quality:
- Create deterministic, disjoint train/validation/test partitions.
- Ensure no sample or generated source-audio item leaks between partitions.
- Train only on the train partition.
- Use validation for checkpoint selection and tuning.
- Keep test untouched until final evaluation.
- Record split indices or stable split fingerprints with the run.

## Duration arithmetic check

At 44,100 Hz, `--duration 0.05` yields 2,205 samples per item. This exceeds the documented 256-sample minimum. The current latent dry-run also chooses 2,205 samples at this duration.

This only verifies the arithmetic floor. It does not prove compatibility with convolutional receptive fields, padding, model-specific minimum sequence lengths, or STFT window/hop requirements. Runtime compatibility remains untested until the missing model/loss/dataset modules exist.

## Scope and status

This addendum is a source-inspection finding, not a training run. No benchmark, checkpoint, held-out score, or successful smoke test is claimed. The three missing functional modules and the unresolved teacher decision remain blockers.
