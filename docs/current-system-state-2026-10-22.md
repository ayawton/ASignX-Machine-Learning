# ASignX - Current system state

## 22 October 2026 | Asia/Singapore | Repository and saved-evaluation review

ASignX recognizes static alphabet hand poses and the dynamic signs J and Z using hand landmarks. The working system retains the older static checkpoint because it scored higher than the recent retraining runs on the same reviewed test data.

| Current item | Verified state |
| Static dataset | 157,734 rows; 24 classes; 1639 session IDs |
| Retained static model | 63,681 parameters; cleaned-test macro F1 63.07% |
| Retained dynamic model | TCN-attention; 296,641 parameters; J/Z |
| Dataset corrections | 2,253 M rows removed; 207 source rows unresolved |
| Software checks | 114 passed, 13 warnings in 25.42s |
| Workspace cleanup | 10 experimental checkpoints removed; 2.37 GiB build/package cleanup |

## How to read this report

Cleaned-test scores and original-test scores describe different evaluation populations. The report labels both explicitly. A higher score after removing wrong labels is not evidence that the saved model learned anything new. Neither offline landmark accuracy nor passing software tests establishes live recognition reliability.

Contents: architecture; data contracts; model architecture; data inventory; source corrections; performance; per-class results; training history; validation; operations and cleanup; priorities; evidence and hashes.

Evidence: Current model comparison, retained checkpoints, correction records and cleanup logs. Full machine-readable hashes accompany this PDF.


---

# 1. Current architecture

The desktop pipeline separates camera capture, hand detection, feature preparation, classification, and display. Offline tools manage collection, import, training, evaluation, and source review.

Camera -> MediaPipe hand points -> versioned features -> static/sequence predictors -> runtime gating -> display

| Layer | Implementation and responsibility |
| Capture and detection | asignx/camera.py and detector.py: OpenCV capture, MediaPipe Tasks hand landmarks, duplicate-hand suppression, handedness stabilization. |
| Recognition | ml/predict.py and predict_sequence.py: preprocessing contracts, confidence thresholds, frame stability, sequence buffer and motion-based routing. |
| Presentation | main.py and asignx/visualizer.py: annotated camera output and recognition state. |
| Dataset tools | Extractor GUI, CSV merger, manual static collection, video/sequence preparation and Dynamic Studio. |
| Training and evaluation | PyTorch trainers, versioned manifests, class metrics, source diagnostics, calibration and review artifacts. |
| Browser integration | web_client/asignx_client_sdk.ts and ONNX export tooling exist; no browser-deployment or mobile performance claim is made here. |

Default combined recognition uses confidence 0.8 for both predictors, 5 stable frames and 10 release frames. The CLI motion threshold defaults to 0.045. These are runtime settings, distinct from the unthresholded offline scores reported later.

Evidence: main.py; asignx/*; ml/predict*.py; web_client/asignx_client_sdk.ts.


---

# 2. Data models and preprocessing

| Record | Contract |
| Landmark / hand / detection | Structured objects describe 21 named points, handedness, detections and hand collections. Detector outputs feed both visualization and feature preparation. |
| Static Sample / CSV | 67 columns: label, session_id, handedness, preprocessing, then 63 landmark coordinates in the code-defined feature order. |
| SequenceSample / NPZ | label, session_id, handedness, preprocessing version, frame-by-feature array and optional JSON metadata. Missing or incompatible contracts are rejected. |
| Split manifest | Dataset SHA-256, labels, seed, preprocessing contract and index lists. Every row must belong to exactly one partition. |
| Checkpoint | Weights, model configuration, label order, preprocessing, training metadata, dataset hash and split assignments. |

## Static transformation

For each hand, subtract the wrist from all 21 xyz points and divide by the largest wrist-to-point Euclidean distance. Flatten to 63 float32 values. Reject nonfinite or degenerate geometry; validate a zero wrist and unit maximum radius. The CSV keeps its original hand convention.

The retained classifier expects left_x_reflection_v1: reflect only x for a Left hand. Keep y and z. This standardizes handedness; it is not palm rotation alignment. Extraction and live detection must use consistent handedness and mirror conventions.

## Dynamic transformation

sequence_anchor_trajectory_v2 uses a common clip origin at the earliest visible wrist and a median visible-hand radius scale. It retains whole-hand motion and two-hand relative placement. Missing-hand slots remain distinguishable; fixed-length interpolation prepares model input. The retained model expects 126 values per frame, not a single static 63-value vector.

Evidence: asignx/models.py; ml/data_io.py; sequence_io.py; preprocessing.py; dataset.py.


---

# 3. Retained model architecture

## Static: kinematic residual MLP

63 coordinates -> 91 features -> 128-wide stage -> 64-wide stage -> cosine-prototype classifier -> 24 class logits. The 28 added features are 15 joint-flexion cosines, 10 fingertip distances and a 3D palm normal. Each stage uses Linear, LayerNorm, GELU and dropout 0.2, followed by a residual MLP block.

The cosine head normalizes the embedding and class vectors, then applies a learned clamped scale. The saved checkpoint has 63,681 parameters. Its labels are A-I and K-Y; J and Z use the sequence branch.

## Dynamic: temporal convolution plus attention

30 frames x 126 coordinates -> 182 kinematic features per frame -> projection to width 128 -> two temporal residual blocks with dilations 1 and 2 -> four-head self-attention with residual normalization -> mean temporal pooling -> 64-dimensional embedding -> cosine-prototype head for J/Z.

The retained sequence model has 296,641 parameters. It is loaded through the compatibility class SignLSTM, but its checkpoint configuration selects backbone=tcn_attention; it is not using an LSTM recurrent backbone. Alternate implementations remain in source code, but their experimental weight files were removed.

## Training algorithms

The retained static checkpoint records 50 epochs, batch size 64, AdamW learning rate 0.002, weight decay 0.0001, label smoothing 0.05, dropout 0.2, seed 42, CPU training and augmentation; its selected epoch is 25. Current trainers support validation-macro-F1 selection with validation loss as a tie-break, cosine learning-rate decay, class weighting and 3D rotation/jitter augmentation. The old checkpoint does not record every newer option, so newer defaults must not be inferred as its historical configuration.

Evidence: Retained checkpoint metadata; ml/model.py; sequence_model.py; train.py; train_sequence.py.


---

# 4. Current dataset inventory

| Measure | Current state |
| Static rows | 157,734 |
| Distinct static labels | 24 |
| Session IDs | 1639 |
| Handedness | Left: 60,218, Right: 97,516 |
| Training / validation / test | 109,223 / 13,791 / 34,720 |
| Original pre-deletion snapshot | 159,987 rows; original test 36,973 |
| Recovered sequence benchmark | 11 clips: training 7, validation 2, test 2 |

## Current class counts

| Class | Rows | Class | Rows | Class | Rows |
| A | 5,853 | I | 7,037 | R | 7,451 |
| B | 5,869 | K | 7,998 | S | 6,241 |
| C | 6,661 | L | 8,252 | T | 7,106 |
| D | 7,086 | M | 3,388 | U | 6,876 |
| E | 6,089 | N | 5,281 | V | 7,410 |
| F | 7,372 | O | 6,221 | W | 6,041 |
| G | 7,534 | P | 5,538 | X | 7,000 |
| H | 7,506 | Q | 5,557 | Y | 6,367 |

## Interpretation

A session ID is a recording/import group, not verified signer identity. Session-separated partitions prevent exact group reuse, but do not establish independence of people, source archives or transformed images. Class imbalance and source-specific appearance can still dominate results.

The weak-class Parquet preparation and its combined experiment CSV are retained as historical data artifacts. They are separate from the current main CSV and must not be silently treated as the active training set. The two historical dataset backups preserve original row identities and reproducibility.

Evidence: Current ml/data/landmarks.csv; reviewed.splits.json in the latest correction directory; sequence_classifier.splits.json.


---

# 5. Source audit and M corrections

The archive (4) M folder contained 7,900 image files, representing 4,876 unique image contents. Source recovery identified the archive-4 M session as ext_c72cf97fc7fa5995335a73a1. Stored rows do not inherently include their original image path, so direct geometric correspondence was required.

| Action / evidence | Result |
| Initial reference correction | Original CSV row 110182, linked to M (100).jpg: one M row removed at the owner's instruction. |
| Full pose-family review | The owner-defined downward three-extended-fingers pose was reviewed across 18 candidate contact sheets; five other-family sheets were also reviewed. |
| Additional confirmed exclusions | 2,252 M rows removed. Matching required all 63 canonical float32 coordinates within absolute 1e-6, with no relative tolerance. |
| Combined removal | 2,253 M landmark rows total. 4,083 source filenames link to those rows because identical images appear under multiple names. |
| Remaining archive-4 M rows | 1,062: 855 matched retained images; 207 could not be linked reliably and were preserved. |
| Scope of supported deletion | The recorded exclusion was M-specific. These records do not substantiate a separate N-class deletion. |

The source ZIP and retained main-CSV rows were preserved. A historical retry without padding-coordinate restoration tested 208 unique images and found no additional exact matches. No deletion was based solely on a classifier prediction or a nearest-pose guess.

The filenames-only list is output/archive4-M-pose-review/removed-image-filenames.txt. Contact sheets, image decisions, unresolved rows, removed-row evidence, full CSV backups and row remaps remain available. Original CSV row numbers and current row numbers must not be interchanged.

Evidence: output/archive4-M-pose-review/*; output/data-repair/latest-correction.json; archive/data-corrections/*.


---

# 6. Recognition performance

The table below concerns the same retained static weights evaluated on two populations. Scores are unthresholded, offline landmark classification, not live camera measurements.

| Metric | Current cleaned test | Original historical test |
| Test sample count | 34,720 | 36,973 |
| Accuracy | 56.62% | 53.19% |
| Macro precision | 65.79% | 63.68% |
| Macro recall | 83.57% | 82.04% |
| Macro F1 | 63.07% | 59.49% |
| Weighted F1 | 59.42% | 56.16% |
| M F1 | 73.06% | 36.73% |
| N F1 | 56.49% | 22.81% |

## Validation and the gap to test

Validation contains 13,791 samples. Retained validation macro F1 is 97.33% and accuracy is 97.64%. The large gap to test is consistent with source/domain mismatch and difficult or disputed examples; it is not proof of one particular preprocessing fault.

## What macro F1 measures

For each class, precision measures how many predictions of that class are correct; recall measures how many true examples are recovered. F1 = 2 x precision x recall / (precision + recall). Macro F1 averages the 24 class F1 values equally; weighted F1 weights them by support. Accuracy counts all correct predictions over all rows.

Removing the 2,253 owner-reviewed M test rows changed the test composition. The 63.07% score does not mean the model improved from training. Because the test has informed development, an untouched benchmark with verified identities is still needed.

Evidence: output/current-model-comparison/comparison.json; original evaluation in output/predeletion-training/.../comparison.json.


---

# 7. Per-class cleaned-test results

| Class | Support | Precision | Recall | F1 |
| A | 235 | 69.58% | 98.30% | 81.48% |
| B | 237 | 93.68% | 100.00% | 96.73% |
| C | 431 | 28.29% | 97.68% | 43.88% |
| D | 472 | 19.79% | 99.58% | 33.02% |
| E | 355 | 63.38% | 99.44% | 77.41% |
| F | 387 | 91.27% | 100.00% | 95.44% |
| G | 4604 | 89.47% | 12.36% | 21.72% |
| H | 515 | 22.19% | 100.00% | 36.32% |
| I | 4373 | 99.78% | 93.55% | 96.57% |
| K | 4999 | 98.09% | 29.79% | 45.70% |
| L | 558 | 22.62% | 99.10% | 36.83% |
| M | 1378 | 94.79% | 59.43% | 73.06% |
| N | 393 | 39.77% | 97.46% | 56.49% |
| O | 4419 | 99.25% | 56.71% | 72.18% |
| P | 317 | 10.46% | 90.85% | 18.76% |
| Q | 3902 | 97.43% | 34.01% | 50.42% |
| R | 520 | 40.07% | 94.23% | 56.22% |
| S | 354 | 62.17% | 98.87% | 76.34% |
| T | 439 | 58.23% | 94.31% | 72.00% |
| U | 397 | 79.27% | 98.24% | 87.74% |
| V | 515 | 13.28% | 99.61% | 23.44% |
| W | 274 | 98.91% | 99.64% | 99.27% |
| X | 4333 | 97.77% | 52.50% | 68.32% |
| Y | 313 | 89.43% | 100.00% | 94.42% |

Lowest class F1 values: P (18.76%), G (21.72%), V (23.44%), D (33.02%), H (36.32%). Prioritize original-image review and targeted collection for these classes; do not remove hard examples solely to raise the score.

Evidence: Existing-model metrics on the reviewed 34,720-sample test set. Supports and confusion matrix are preserved in current-model-comparison/comparison.json.


---

# 8. Retraining and model selection history

| Run / evaluation | Validation macro F1 | Test macro F1 | Test accuracy |
| Retained old model / cleaned test | 97.33% | 63.07% | 56.62% |
| Recent retraining / cleaned test | 97.72% | 57.77% | 49.57% |
| Retained old model / original test | 97.33% | 59.49% | 53.19% |
| Recent retraining / original test | 97.72% | 54.13% | 46.56% |

Two 50-epoch retraining runs used the reviewed and pre-deletion snapshots with preserved partitions. Both selected epoch 43 by validation macro F1 and produced identical learned weights; this equality was checked before the experimental checkpoints were removed. Deletion affected only test rows, so the training and validation inputs were the same.

The newer model improved validation slightly but underperformed the older model on both versions of the test. The older canonical checkpoint remains the selected application model. Its original training dataset hash and split manifest were not rewritten to match the cleaned CSV.

## Artifact disposition

Ten unused experimental checkpoints were deleted at the owner's request: five static F1 variants, two dynamic variants, the weak-class candidate, and the two recent retraining checkpoints. Their metrics, predictions, manifests and cleanup hashes remain. Reports describing those models are historical evidence, not promises that the checkpoint files are still runnable.

## Dynamic results require a separate qualification

The retained J/Z model scored 100% test macro F1 on only two clips, one per class, from the recovered 11-clip benchmark. That is insufficient for a reliability claim. Existing temporal diagnostics also flag possible shortcuts: tiny test performance staying unchanged after motion removal needs a much broader sequence benchmark.

Evidence: Retained model metadata; cleaned/pre-deletion comparisons; output/model-cleanup; recognition-reevaluation/summary.json.


---

# 9. Validation and current checks

| Check | Current evidence / limit |
| Fresh software suite | 114 passed, 13 warnings in 25.42s. Warnings concern the existing tracing/export paths; see the saved log. |
| Static data contract | 63 finite float32 features; wrist-centering and unit-radius checks; required preprocessing and source groups. |
| Partition integrity | Every row assigned once; each split contains all labels; session leakage, exact canonical feature leakage and conflicting exact labels rejected. |
| Import protections | Schema checks, archive path checks, byte preservation, duplicate/conflict detection, output locks, source grouping and audit records. |
| Source-image exclusion | Image-led decisions plus coordinate correspondence; retained rows byte-identical, only M rows excluded, source ZIP hash unchanged. |
| Retained extractor package | 2,016 package files matched the ZIP by hash. Packaged smoke import completed, skipping two label conflicts and a blank no-hand image. |

## What these checks do not establish

Passing integrity checks cannot verify every linguistic label, signer identity or subtle near-duplicate. The package smoke test checks startup and a limited import path; it does not quantify landmark quality on representative images. Offline predictions omit detector failures, unknown-sign rejection, transitions, repeated gestures and live end-to-end latency.

The packaged extractor is the newest retained packaged release, not a statement that its embedded code was rebuilt after every later source-code repair. The smoke audit records its preprocessing identifier. Preserve the versioned contracts and re-extract into a separate file when comparing package and current-source behavior.

Evidence: output/system-cleanup/current-report-tests.log; package-verification.json; smoke extraction audit; ml/dataset.py; tests/test_extractor_integrity.py.


---

# 10. Operations and workspace cleanup

| Retained component | Location / use |
| Static checkpoint | ml/models/sign_classifier_canonical.pth |
| Dynamic checkpoint | ml/models/sequence_classifier.pth |
| Current extractor package | output/extractor-reviewed/ASignX_DataExtractor |
| Matching distribution ZIP | output/extractor-reviewed/ASignX_DataExtractor.zip |
| Dynamic Studio | dist/ASignX_DynamicStudio.exe |
| Current comparison | output/current-model-comparison/report.md |

## Launch commands from the project folder

.\ASignX_TestRecognition.bat

.\ASignX_DataExtractor.bat

.\ASignX_DynamicStudio.bat

The recognition launcher selects the retained static and sequence checkpoints. The extractor launcher selects the reviewed package and falls back to Python source if it is missing. The Dynamic Studio launcher keeps its executable-first behavior.

## Removed and preserved

Build/package cleanup removed 2.37 GiB across 14 directories: three older extractor packages, build intermediates, five temporary build folders and caches. The launcher and guide were updated; build outputs and .venv-py314 were added to ignore rules. Git's index was not modified.

Both active model hashes, the main dataset, historical backups, reviewed extractor, its ZIP and Dynamic Studio executable were checked unchanged. Audit/recovery scripts and temporary dependency installations were kept. Reports and manifests for deleted experiments remain intentionally historical.

Evidence: ASignX_*.bat; .gitignore; docs/extractor-guide.md; output/system-cleanup/cleanup-record.json; output/model-cleanup/*.


---

# 11. Remaining issues and next steps

| Priority | Action | Evidence of completion |
| 1 - Label evidence | Review weak-class images by original source; resolve the 207 unmatched archive-4 M rows without deleting by prediction alone. | Recorded image identity, reviewer decision, reason and traceable row correction. |
| 2 - Independent benchmark | Collect new signers and recording sessions across cameras, lighting, backgrounds, distances and hand orientations. Reserve an untouched global identity split. | All static labels and J/Z covered; signer/source groups stay in one split across labels. |
| 3 - Targeted training data | Add verified examples for weak classes and confusing pairs. Preserve pose diversity and audit near-duplicates across partitions. | Source-balanced learning curves and per-class precision/recall on validation; one final held-out test. |
| 4 - Dynamic evidence | Expand J/Z clips, wrong attempts, idle frames and transitions; assess gesture events and motion-removal diagnostics. | Per-class support large enough to report uncertainty; demonstrated temporal benefit. |
| 5 - Pipeline parity | Compare current-source and packaged extraction on the same verified images; track mirror/handedness, padding and preprocessing versions. | Same feature contract or documented migration into a separate dataset. |
| 6 - Runtime benchmark | Measure live detector failures, end-to-end latency, false triggers, coverage and calibrated unknown rejection. | Reproducible live sessions with device and environment metadata. |

Keep the current model until a candidate passes a predefined validation gate and then improves on an independent benchmark. Do not shrink the test set, change labels from predictions, or repeatedly tune against the historical test to create a higher headline score.

Evidence: Current per-class results, unresolved-row audit, historical source diagnostics and dynamic benchmark support.


---

# 12. Evidence, identity and reproducibility

This report captures the repository state on 22 October 2026 in Asia/Singapore. Recognition scores were read from saved evaluations tied to explicit dataset/model identities. Software tests were rerun for this report. Models and datasets were not changed to produce the document.

## Current cleaned static CSV

4a160bec73556408c12f9457577c765c
89e9f8cf09eae3d658fe39d3064f58cd

## Original static dataset

17249bc657b7ed52f1d6a39f5d3bbbf6
b8d2445efdab97f64094df68855647db

## Retained static checkpoint

98559a10c553c23d2c6aad227f52416b
59ad92483bf4f079cfca5eb512bf5b66

## Retained sequence checkpoint

1e2228a5970d18b19a25f62cb76fc560
aa4c5e33962a8bc2a362a87309db2ec9

## Primary local evidence index

Current evaluation: output/current-model-comparison/comparison.json

Image decisions and filenames: output/archive4-M-pose-review/

Correction chain and backups: archive/data-corrections/

Latest correction summary: output/data-repair/latest-correction.json

Original-test historical scores: output/predeletion-training/20261021T162040Z/comparison.json

Removed model inventory: output/model-cleanup/

Package/build cleanup and fresh tests: output/system-cleanup/

The companion .evidence.json records source hashes, model configurations, current counts and the fresh test result. The editable Markdown and builder are under docs/. Retained historical model manifests are bound to their original data hash: using them directly with the cleaned CSV should fail strict identity checks, not be bypassed by overwriting metadata.
