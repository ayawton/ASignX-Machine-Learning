# ASignX whole-system diagnosis — 24 October 2026

> Provenance update: [recovered import archives, local capture batches, matching images and source-conditioned errors](../source-recovery/report.md). The previously unresolved hashed import bucket is now linked to archive (4); signer identities remain a separate limitation.

> Subsequent work: [implemented live-state fixes, regression tests and benchmark readiness](fixes.md). The reproduced source behaviors below describe the pre-fix state; model-score limitations remain relevant.

## Assessment

The project has working detection, collection, training, and inference infrastructure, but its current evaluation does not establish reliable recognition for new users. The largest measured problem is poor performance on a large subset of imported test samples despite high validation scores. There are also reproducible live-camera and browser state problems. A larger network or another training run alone is not an established remedy.

This diagnosis reads the current source, checkpoint metadata, current dataset and existing fixed-benchmark predictions; runs the full test suite; and reproduces specific behaviors with synthetic inputs. It does not capture new human signing videos, retrain, change labels, or replace application models. Findings below distinguish observations from possible explanations.

## 1. High priority: validation is a weak guide to the difficult test data

| Model | Validation accuracy | Validation macro F1 | Historical test accuracy | Historical test macro F1 |
|---|---:|---:|---:|---:|
| Retained application | 97.64% | 97.33% | 56.62% | 63.07% |
| Latest full retraining | 97.70% | 97.57% | 52.07% | 59.30% |

These use the same 13,791 validation and 34,720 reviewed historical test rows. The latest model's slightly higher validation F1 accompanies lower test F1. Selecting a checkpoint by that validation set therefore does not reliably select for this difficult test set.

The gap is **not explained by letter proportions alone**. Applying the test set's letter weights to the retained model's validation per-letter recall gives 98.34% expected accuracy, versus 56.62% actually observed on test. This is a diagnostic reweighting calculation, not an estimate of deployment accuracy.

Session-name groups expose where the failure concentrates:

| Session-name hint | Validation rows | Retained validation accuracy | Test rows | Retained test accuracy | Latest retrained test accuracy |
|---|---:|---:|---:|---:|---:|
| `legacy:archive5` | 3,581 | 99.97% | 2,838 | 99.93% | 100.00% |
| `legacy:archive3` | 6,045 | 97.09% | 6,564 | 97.55% | 98.43% |
| Unresolved import identity | 4,165 | 96.45% | 25,318 | 41.15% | 34.68% |

The unresolved group accounts for 72.92% of test but 30.20% of validation. These are naming hints, **not verified signer or source identities**. Its validation and test rows are different subsets; this table does not imply identical examples changed performance between partitions. It identifies a large, poorly understood region requiring original-source review.

Possible explanations include source/viewpoint differences, label conventions, incorrect labels, and related examples crossing splits. This diagnosis does not establish their individual contributions. Review original evidence before relabeling or deleting anything. Do not repeatedly tune against the historical test; obtain a new independent benchmark.

Evidence: [source breakdown](source-performance.csv), [dataset and metric evidence](evidence.json), [latest fixed-benchmark comparison](../full-static-retraining/20261023T195828Z/comparison.json).

## 2. High priority: split identity and score comparability

The current CSV contains 159,534 samples, 24 static letters, 1,648 session strings, 62,018 Left samples and 97,516 Right samples. The reviewed historical partitions plus all new samples give 111,023 training rows, 13,791 validation rows and 34,720 test rows. The new M data contain 1,800 left-hand frames across nine sessions; frames and sessions are not counts of independent signers.

The default split code keeps session strings together and rejects exact canonical duplicates across splits. These are useful checks. However, sessions must each have only one label, so the same signer can occur under separate sessions for multiple labels and cross partitions. Imported grouping strings do not prove actual recording independence. Global identity grouping is implemented in `ml/benchmark.py`, but the inspected training configurations have `group_plan: null`.

Letter counts are also very different between partitions. Validation includes 4,401 W samples (31.91%); test includes 274 W samples. Test instead includes 4,999 K, 4,604 G, and 4,419 O samples. Macro F1 helps equalize class influence but cannot repair differences in source, signer, pose coverage or label quality.

There is a separate manual checkpoint at `ml/models/manual_runs/20261022T071741Z-87dde74ec0df/sign_classifier.pth`. Its dataset hash matches today's CSV, but its generated split differs from the fixed historical benchmark. Its own reported validation/test macro F1 values are 85.79%/74.23%. Those numbers cannot be compared directly with the retained model's historical scores: **1,886 historical test rows and 784 historical validation rows occur in that manual model's training set**. Do not evaluate it on the historical test and call all rows held out.

Evidence: `ml/dataset.py:27`, `ml/benchmark.py`, `ml/manual_training_gui.py:377`, [split compatibility](split-compatibility.json).

## 3. High priority: cached camera frames can count as fresh observations

`asignx/camera.py:213` waits for a frame event with a 0.1-second timeout, but after a timeout it still reads and yields the last cached frame while capture remains open. The capture worker also leaves the last successful frame available after failed reads.

A synthetic open-camera stream with no new-frame event yielded the same cached frame twice. Downstream recognizers assign processing timestamps, so a sensor stall can appear to be a fresh, perfectly held pose. This can advance static confirmation or conceal missing dynamic observations. Sequence timestamp-gap checks cannot detect every stall if the stream keeps supplying stale pixels with fresh processing times.

Recommended repair: carry capture sequence IDs and capture timestamps, yield only fresh frames, and explicitly report stale capture/missing observations. Verify with a stalled-camera replay, not just a blank image. Frequency on actual hardware remains unmeasured.

Evidence: [reproduction](evidence.json), `tmp/diagnose_system.py`, `asignx/camera.py:159` and `:213`.

## 4. High priority: handedness recovery can override strong detector evidence

`asignx/detector.py:194` predicts a previous Left hand's centroid and forces a nearby non-Left detection to Left when no Left label exists. This branch does not check detector confidence. A synthetic Right detection with confidence 0.99 was changed to Left when it appeared at the previous Left track's location.

This demonstrates an unconditional override, not a measured real-world error rate. If a different physical hand replaces the tracked hand, the heuristic can assign the wrong identity. A wrong hand label changes static x-reflection, temporal state ownership and dynamic Left/Right slots. The duplicate suppression also uses fixed image-space wrist/centroid distances and overlap; close or crossing real hands need explicit replay coverage.

Recommended repair: track identities separately from anatomical handedness, qualify reassignment using confidence and motion evidence, and test hand replacement, crossings and partial occlusion.

## 5. Confirmed browser helper problems and integration risks

The SDK exports preprocessing, coaching and stabilization helpers; it is not itself a complete MediaPipe + ONNX browser application.

Reproductions using the actual SDK source found:

1. **Stale stable label on a new candidate.** After A is confirmed, the first B prediction can return `predictedLabel: B` and `stableLabel: A`. Python's stabilizer returns no stable label until B is confirmed. This is a behavioral mismatch that can show stale feedback.
2. **Missing-hand state remains stable temporarily.** `markHandMissing()` retains `stableLabel` until the release count. A caller displaying that state can continue showing a hand's old confirmed sign after disappearance. Separate display policy from current observation validity.
3. **Coaching progress carries across target changes.** With a two-frame hold requirement, one accepted A attempt followed by one accepted B attempt passes B. Callers must currently reset on every target change; the class does not enforce it.
4. **Missing references claim finger alignment.** Without a centroid, finger error defaults to zero, so every finger is marked aligned. Return “reference unavailable” rather than positive coaching feedback.
5. **Hand state is not partitioned internally.** One `ASignLiveStabilizer` receiving both hands at the same frame timestamp throws a timestamp error; different timestamps can mix histories. This is an integration requirement: use one instance per hand or implement per-hand state explicitly. The `hand` argument alone does not provide isolation.

Also, TypeScript metadata declares required `hidden_dim`, whereas the static exporter emits `hidden_sizes`. The shown helpers do not use that field, but typed consumers need a consistent contract. Exported static graphs already contain the model's kinematic enrichment: feed externally normalized/canonicalized 63D inputs, not externally enriched 91D features.

Evidence: [SDK probe results](sdk-reproductions.json), `tmp/diagnose_sdk.mjs`, `web_client/asignx_client_sdk.ts:229`, `:320`, `ml/export_onnx.py:36`.

## 6. Training capabilities and what the regressions establish

`ml/train.py:183` constructs a fresh `SignMLP`. It does not expose an initial-checkpoint/warm-start option; its checkpoint loader is used after training to restore the selected result. The GUI invokes this same trainer. Fine-tuning the whole model has therefore **not** been tested in the recent comparisons.

Training from scratch may contribute to variation, but it is not a proven explanation for the overall regression. The comparisons change added rows, batches, augmentation assignments and inverse class weights together. Single-seed runs do not isolate those mechanisms.

A whole-model fine-tuning experiment should start with the retained checkpoint, verify label order and preprocessing, include training examples from all letters, use a lower learning rate, and retain the starting checkpoint as an eligible baseline. If preserving old behavior is essential, compare ordinary fine-tuning with teacher-output regularization or parameter constraints. Select using representative validation and predefined overall/per-letter regression limits; reserve independent test data for the final comparison. No guarantee of improvement is justified yet.

## 7. Dynamic recognition and unknown gestures remain unvalidated

The saved dynamic model predicts J/Z. The matching recovered benchmark has 11 clips: seven training, two validation, two test. Its 100% test F1 comes from one clip per class and is insufficient evidence of general movement recognition.

Training/prepared clips are duration-resampled complete gestures. Live prediction uses a fixed sliding window, permits inference when half full with zero padding, and uses motion gating plus stabilization. Complete clips and partial rolling windows are different inputs; segmentation, speed variation, hand visibility and static-to-dynamic transitions need continuous-stream replay. Timestamp sampling and missing-data checks are implemented, which is useful but not an end-to-end accuracy demonstration.

Static softmax thresholding is implemented; it does not establish rejection of arbitrary non-sign poses. At threshold 0.8 the retained model accepts 85.53% of known-sign historical test rows, with only 60.67% accepted-only accuracy. The existing evaluation does not measure false activations on idle hands, transitions or unknown gestures. Collect negative streams and report false emissions per minute, missed signs, duplicate emissions and latency.

Evidence: [dynamic benchmark](../recognition-reevaluation/dynamic.json), `ml/predict_sequence.py:139`, `ml/predict.py:44`, `ml/predict.py:275`.

## 8. Deployment and artifact identity

`ASignX_TestRecognition.bat` explicitly loads `ml/models/sign_classifier_canonical.pth` and `sequence_classifier.pth`. Training a candidate or the manual GUI does not update this launcher. The retained static checkpoint was trained against an older dataset hash; this does not prevent inference, but its original manifest cannot directly evaluate today's changed CSV. Use identified snapshots and compatible remapped partitions.

Several historical reports describe different data compositions. Filtering disputed test labels changed the test from 36,973 to 34,720 rows; this changed reported scores without changing model weights. The latest candidate reports explicitly retain fixed rows. Keep the model, dataset hash, split identity and threshold alongside every score.

The exporter catches every ONNX export exception and falls back to TorchScript. A returned export bundle therefore does not guarantee a browser-loadable ONNX model. Export tests check artifacts, not full browser prediction and temporal parity. Verify ONNX existence, runtime loading and numerical predictions in the intended browser before deployment.

## Verification and recommended order

**114 tests passed, 7 warnings.** The initial direct run could not create/reopen pytest's Windows temporary directories under the sandbox. The final runner changes only newly created temporary-directory mode 0700 to the normal inherited mode and sets the project import path; application/test source and existing ACLs are untouched. Logs: [tests.log](tests.log); runner: `tmp/diagnose_test_runner.py`. Passing tests do not invalidate the separately reproduced behaviors above.

1. Fix and replay-test fresh-frame handling, handedness reassignment, and SDK state/reference behavior.
2. Recover real source/signer provenance and visually audit samples in the difficult import groups; preserve originals and record review decisions.
3. Establish representative signer/source-disjoint validation and a new untouched test; retain the historical benchmark as development evidence.
4. Test whole-model fine-tuning against the retained starting checkpoint, with all-letter replay and explicit regression limits. Do not replace the current model based on validation improvement alone.
5. Collect independent dynamic clips and continuous positive/negative streams; verify browser/Python/export parity and end-to-end latency.

Keep the retained application model for now. The observed evidence supports fixing data/evaluation and live-state reliability before assuming network size is the bottleneck. The exact cause of each letter's regression remains unproven.
