# Preparing a signer-separated recognition benchmark

Use an anonymous ID for a real person you can consistently recognize across their recordings. Their name is unnecessary. Do not assign a new signer ID to each imported session: several sessions may belong to one person. A download archive is a source, not a person.

## Current data and readiness

Run:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.benchmark_readiness --data ml/data/landmarks.csv --output output/signer-benchmark
```

This creates `identity-plan.csv` if absent and updates `readiness.json`. It preserves an existing plan, dataset and checkpoints. Exit code 2 means the benchmark is not ready, not that the command crashed. Unknown identities start blank with status `unknown`; no independent people are invented.

For sessions whose real grouping can be verified, fill:

- `signer_id`: a stable anonymous person ID, reused across every letter, hand and visit.
- `group_id`: normally that same person ID.
- `recording_id`: the original recording/visit family, when known.
- `source_id`: original source family, when known; never a fabricated signer.
- `device`: collection device identifier, when known.
- `split`: `train`, `validation` or `test`. Assign the person to one split for every session and label.
- `identity_status`: `verified` only when the grouping is supported by records.
- `identity_evidence`: where the grouping was verified, such as a collection register. It need not contain personal information.

The remaining columns summarize the actual stored samples. They are informational; the audit derives coverage from the dataset, not edited count cells. Re-run the audit after edits. It checks cross-label identity/source/session leakage, exact canonical duplicates, every letter and both hands in each split. Collection targets require at least three train signers and two validation and test signers per letter. These are minimum planning targets, not proof of deployment reliability; add signer, viewpoint, device, lighting and performance diversity.

## When existing signer identities are unknown

Keep them unknown. The current public-source data cannot honestly establish signer separation until grouping evidence is recovered. Source-only or session-only experiments may remain useful development diagnostics, but label them accordingly. An invented list of distinct anonymous people would not make them independent.

The practical alternative is a new supervised collection:

1. Recruit distinct people into train/validation/test pools before recording. Assign each actual person one anonymous ID; record them in a local collection register.
2. Collect every static letter (A–I, K–Y) from each person with both hands where appropriate, across at least two visits and varied conditions. Keep every visit from that person in their assigned split. Verify the sign convention visually before extracting samples.
3. Keep a register linking exported session IDs to anonymous signer IDs and visits. For the static studio, session IDs are retained in its batch manifests; copy the verified mapping into the plan. Adding frames does not add independent signers.
4. Collect J/Z as complete independent takes using the dynamic tools' signer, recording, device and split fields. Include idle hands, transitions, wrong paths and unknown gestures as separately labeled stream-evaluation material.
5. Keep the new test untouched while selecting settings. Capture timestamps, emitted labels and expected sign events for continuous-stream evaluation. Report event recall, false emissions per minute, duplicates and latency as well as letter metrics.

The retained model was trained on public data with incomplete identities. Its training-person overlap with a proposed test cannot be ruled out merely by renaming sessions. Document this limitation when comparing pretrained models.

## Automated source-only experiment

The completed recovery pass in [the source report](../output/source-recovery/exhaustive/report.md) checked 89,774 color/static images, added image evidence for 27,016 stored rows and applied conservative source co-location constraints to all 1,648 sessions. Partial-session matches constrain co-location; they do not establish the exclusive origin of every row or verify a person.

A separate [50-epoch static candidate](../output/source-separated-training/20261024-source-holdout/report.md) uses 122,708 supported rows: archive (4) plus local captures for training, archive (5) for validation, archive (3) for testing. All partitions contain all 24 static letters and pass session/canonical-duplicate checks. The other 36,826 rows remain in the original dataset and are excluded only from this experiment. Signer independence and balanced handedness coverage remain unverified.

The candidate starts from scratch because existing model weights were exposed to some rows of this new holdout. It is a source-transfer development baseline, not an uncontaminated comparison against the deployed model. Dataset, active checkpoint and J/Z weights remain unchanged. The experiment's protocol, row ledger, histories, predictions and independent verification are linked from its report.

## Training after the plan is ready

After the readiness audit passes, use a fresh output directory and the exact plan:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.train --data ml/data/landmarks.csv --group-plan output/signer-benchmark/identity-plan.csv --output output/verified-benchmark/model.pth
```

This command uses the data path supplied above; if you collected a new dataset, pass that new dataset to both audit and training. Do not overwrite the retained application model. Training and evaluation must use the same dataset hash, labels, preprocessing and split manifest.

Whole-model fine-tuning should use representative validation, all-letter training replay, a lower learning rate and explicit regression limits. The trainer supports an explicit starting checkpoint:

```powershell
.\.venv-py314\Scripts\python.exe -m ml.train --data ml/data/landmarks.csv --group-plan output/signer-benchmark/identity-plan.csv --initial-checkpoint ml/models/sign_classifier_canonical.pth --output output/verified-finetuning/model.pth --learning-rate 0.0001 --epochs 10 --no-test
```

Use this command only with a reviewed, ready identity plan and matching model architecture, labels and preprocessing. The starting checkpoint remains eligible as epoch 0, selected against later epochs by validation macro F1 then loss. Source and output paths must differ. Optimizer state is initialized afresh; this is fine-tuning, not resuming an interrupted optimizer. Passing a current model as an output filename alone still starts from scratch and does not initialize its weights. The readiness audit does not run training.

A separate historical-split fine-tuning trial was explicitly requested while signer identities remained unknown. Its results are diagnostic and do not change the readiness assessment or establish signer independence. See `output/whole-model-finetuning/latest-run.json` for its artifacts.

## Live-state integration changes

- Threaded `WebcamStream.read()` consumes each new capture once. `False, None` can mean no fresh frame is available yet; polling clients should wait, not reuse old pixels. After two seconds without a successful capture, reads/streams raise `CameraError` so a disconnected sensor cannot hang the preview indefinitely. `frame_timestamp` is the monotonic timestamp of the returned capture.
- Main static/dynamic recognition uses capture timestamps and restarts confirmation after a long frame gap.
- `ASignLiveStabilizer` keeps separate Left/Right state. Use `markHandMissing(hand)` for one missing hand, or omit the argument when neither is visible. `reset(hand)` clears one state; `reset()` clears both.
- Browser `stableLabel` becomes null when a new label is not confirmed or the hand disappears. Duplicate-emission release remains separate from observation validity.
- Coaching automatically resets hold progress on a target change. Missing centroids return `referenceAvailable: false`, `meanError: null` and `isAligned: null`; render “reference unavailable,” not a green alignment indicator.
- Export metadata uses `hidden_sizes`; the old `hidden_dim` SDK field remains optional for compatibility. Static ONNX inputs remain 63 normalized, canonicalized coordinates.

Source code fixes do not rebuild existing packaged EXEs or deploy a browser application. Rebuild distribution artifacts after source verification and before shipping them.
