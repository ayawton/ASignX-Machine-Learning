# Improving and measuring F1

Source recovery update: [data repair investigation](data-repair.md) documents the recovered archive behind 96,145 hashed-import rows, the geometrically matched original image for row 110182, and the new import/manifest safeguards. Historical metrics and original splits are preserved. The 24 recovered source sessions are archive groups, not verified signers.

Implemented on 20 October 2026. Existing checkpoints and `ml/data/landmarks.csv` are preserved. These tools expose errors, support controlled experiments and prepare independent collection; they cannot create new real signers or recover unknown provenance.

## Review the current failures

Open `output/f1-review/baseline/review.html` locally. Filter landmark sketches by class, session or row. `predictions.csv` includes every test prediction; `review-decisions.csv` contains pending human review items. Sketches show stored coordinates, not original images. Corrections require verified original evidence; no rows have been silently removed or relabeled.

The baseline test macro F1 is 59.49%. Accuracy on rows with the legacy archive5 hint is 99.93%, versus 37.82% for hashed-import rows. These hints are diagnostic strings, not verified signer identities. The largest confusion counts are Kâ†’V (3,304), Mâ†’N (2,437), Qâ†’P (2,280), Gâ†’L (1,870) and Gâ†’H (1,711).

Reproduce the audit:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.audit_f1
.\.venv-py314\Scripts\python.exe -m ml.audit_f1 --split validation --output output/f1-review/baseline-validation
```

## Controlled training

Both trainers now select checkpoints by validation **macro F1**, breaking ties with validation loss. `--selection-metric loss` reproduces the older selection policy. `--class-weighting` accepts `inverse`, `sqrt` or `none`. Static `--augmentation-scale` scales rotation and jitter. `--no-test` avoids evaluating the test set during tuning. Invalid or changed existing manifests fail explicitly; choose a new output path to create a reviewed new benchmark.

Run the paired five-variant experiment:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.experiment_f1 --epochs 12
```

Variants change weighting, augmentation strength or label smoothing one at a time. Each uses the same seed, frozen split, 12-epoch schedule and batch size 512. Selection uses validation only; the winner is tested once after selection. These short CPU experiments are not directly equivalent to the older 50-epoch, batch-64 production run. Results and candidate checkpoints live under `output/f1-review/experiments/`; production files are never overwritten. The command resumes completed variants and refuses to reuse different configurations at the same paths.

Compare per-class precision/recall as well as macro F1. A validation improvement can fail on a different source. Do not repeatedly select variants by the test score. A fresh untouched benchmark is still required after this historical test has informed development.

## Real global groups

`output/f1-review/baseline/identity-plan-template.csv` lists every current session with identity fields blank. Fill them only from verified provenance. Required columns are `session_id,group_id,split`; optional identity columns are `signer_id,recording_id,source_id,device`. Each session needs exactly one row. A global group, signer, recording or supplied source must stay in one split across **all** labels. Each split must contain all labels. Use globally unique recording IDs, such as `person01-visit01`, rather than reusing `visit01` for unrelated people.

```powershell
.\.venv-py314\Scripts\python.exe -m ml.train --group-plan path/to/reviewed-identities.csv --output ml/models/new-benchmark/model.pth --no-test
.\.venv-py314\Scripts\python.exe -m ml.train_sequence --group-plan path/to/reviewed-sequence-identities.csv --output ml/models/new-sequence-benchmark/model.pth --no-test
```

The default historical session splitting remains available for compatibility and diagnostic comparisons. Metrics explicitly label it as not signer-independent. The new group-plan loader rejects blank identities and identity leakage; it does not infer identities from a class name or session suffix.

## Confidence rejection

Fit thresholds from validation predictions only:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.calibrate_f1 --predictions output/f1-review/baseline-validation/predictions.npz --output output/f1-review/baseline.calibration.json
```

Rejected known signs count as false negatives. A fitted threshold is bound to the checkpoint SHA-256 and label order. Optional negative prediction files must contain validation logits from the same model, matching labels/hash, and targets of `-1`; provide them using `--negative-predictions`.

The baseline's validation-selected threshold is 0.535, with validation macro F1 97.49%. On the historical test it scores 61.54%, versus 63.06% at the current default 0.8. Neither has been tested on representative unknown gestures. The default remains 0.8. The fitted JSON is an experimental artifact, not a recommendation to lower the production threshold.

An explicit `main.py --sign-calibration path/to/file.json` enables a chosen matching artifact. It is never applied automatically. For unknown rejection, collect independent idle/transition/wrong-attempt examples; do not count all-zero feature vectors as realistic negatives.

## Dynamic data and collection

The portable app's `dist/data/sequences` contains 16 clips: 11 J and 5 Z. An exact 11-clip subset reproduces the saved checkpoint's dataset SHA-256 and has been copied, without changing the originals, to `ml/data/recovered_sequence_baseline`. Its two-clip test is reproducible but far too small for reliability claims; no signer identities are recorded. The earlier report missed these files by excluding packaged directories. Recovery hashes are at `output/f1-review/dynamic/recovery.json`.

Generate the checklist and readiness report (automatically discovers the portable data folder):

```powershell
.\.venv-py314\Scripts\python.exe -m ml.collection_plan
```

The checklist proposes 10 people, two visits, varied J/Z takes and separate negative streams. These are planned slots, not recorded samples or a statistical guarantee. Reserve people 1â€“6 for training, 7â€“8 for validation and 9â€“10 for final testing before recording. Replace planning placeholders with actual consistent pseudonymous identities. Keep negative recordings in a separate folder so J/Z training does not accidentally treat them as ordinary target classes.

Reproduce the saved sequence evaluation with `.\.venv-py314\Scripts\python.exe -m ml.audit_sequence`. A paired 50-epoch LSTM/TCN-attention diagnostic comparison is saved under `output/f1-review/dynamic`. Both use the recovered original split and validation-only selection. This tiny comparison is not a reason to replace the production model.

Temporal diagnostics leave the original checkpoint at 100% on its two test clips even after time reversal, shuffling, repeating the last frame or removing wrist travel. This benchmark therefore does not demonstrate that the model recognizes the actual movement. The checklist includes matching-handshape holds and incorrect J/Z paths as hard negatives. Review these attempts with a knowledgeable signer. Five additional J clips outside the saved dataset are all accepted as J, but their signer identities are unknown and they provide no additional Z/negative evidence.

Dynamic Studio now has a **Recording identity** tab for person, recording visit, source, device and reserved split. Source launch is `.\.venv-py314\Scripts\python.exe -m ml.dynamic_studio_gui`; existing packaged executables require rebuilding to include the new fields. CLI collection also supports metadata:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.collect_sequence --label J --takes 5 --exposure auto --signer-id person01 --recording-id person01-visit01 --device-id laptop01 --benchmark-split train
```

Rerun `ml.collection_plan` after capture to extract an identity CSV from `.npz` metadata and check benchmark readiness. Imported videos still require a reviewed identity plan; their filenames are not automatically trusted as person IDs.

Live inference now samples by timestamp at the saved training FPS, with a documented 15 FPS fallback for old checkpoints. It rejects poor visibility, long tracking gaps and all-zero windows; camera stalls clear stale sequences. Motion routing considers both hand slots, uses elapsed time with a reference-30-FPS threshold, and allows a short post-motion interval for final decisions. Python and TypeScript use the same motion calculation; the browser clears stale labels and releases repeated signs after rejected predictions. These changes have replay/regression coverage, but real continuous-stream F1 and false emissions per minute still require recordings.

## Verify and update the build report

```powershell
.\.venv-py314\Scripts\python.exe -m pytest -q
.\.venv-py314\Scripts\python.exe docs/build_system_state_report.py
```

Update `docs/system-state-current.md` with measured results before rebuilding. Archive the previous report and matching artifacts. The report builder refreshes hashes, not the interpretation of new experiments.

## Completed 20 October experiment result

Validation selected `no_smoothing` (97.31% macro F1). Its unthresholded test macro F1 is 54.98%, below the existing checkpoint's 59.49%. The existing checkpoint and threshold 0.8 are retained. These five 12-epoch screens did not deliver a reliable F1 gain. Original-image review, source coverage and a genuinely independent benchmark are the next required work. Full results are in `output/f1-review/experiments/comparison.json`.
