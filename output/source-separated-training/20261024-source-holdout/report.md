> Model cleanup: unused candidate weights were deleted at the owner's request on 24 October 2026. Historical metrics, predictions and source data remain available. The deployed static and J/Z models were retained. See [cleanup inventory](../../model-cleanup/20261024/report.md).

> Follow-up: [direct current-model comparison on identical rows, with original-training exposure audited](current-comparison.md).

# Source-separated static model experiment

One new 50-epoch candidate trained from scratch across all 24 static letters. Archive (4), including its archive.zip alias, and the local M captures are training-only; image-recovered archive (5) rows are validation-only; image-recovered archive (3) rows are test-only. The allocation was written before training and was not chosen using candidate test scores.

Included 122,708 source-supported rows; excluded 36,826 unresolved-source rows from this experiment only. Train 95,692; validation 11,281; test 15,735. Every partition contains all 24 letters. Source families and original sessions do not cross partitions, and the existing canonical duplicate/conflicting-label validator passes.

| Metric | Validation: archive (5) | Test: archive (3) |
|---|---:|---:|
| Accuracy | 84.84% | 62.84% |
| Macro F1 | 83.08% | 59.79% |

Selected epoch **11 of 50**, using validation macro F1, with loss as the tie-breaker. Test inference ran only after checkpoint selection. No current model weights were used to initialize this candidate, because the retained model has already trained on some rows of the new holdout.

| Letter | Validation rows | Validation F1 | Test rows | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 271 | 98.01% | 425 | 87.76% | 80.94% | 84.21% |
| B | 447 | 100.00% | 267 | 34.65% | 59.18% | 43.71% |
| C | 470 | 99.47% | 823 | 90.06% | 18.71% | 30.99% |
| D | 494 | 96.67% | 672 | 88.10% | 46.28% | 60.68% |
| E | 471 | 100.00% | 555 | 38.67% | 78.38% | 51.79% |
| F | 282 | 100.00% | 633 | 92.06% | 98.89% | 95.35% |
| G | 603 | 0.00% | 1,114 | 75.91% | 89.41% | 82.11% |
| H | 572 | 84.12% | 1,145 | 75.17% | 99.13% | 85.50% |
| I | 485 | 67.45% | 762 | 80.55% | 84.78% | 82.61% |
| K | 535 | 99.25% | 840 | 29.58% | 37.26% | 32.98% |
| L | 481 | 100.00% | 716 | 98.30% | 96.93% | 97.61% |
| M | 406 | 0.00% | 637 | 92.74% | 60.13% | 72.95% |
| N | 592 | 79.84% | 673 | 38.49% | 28.83% | 32.97% |
| O | 386 | 89.35% | 599 | 72.23% | 55.59% | 62.83% |
| P | 562 | 62.11% | 462 | 47.27% | 5.63% | 10.06% |
| Q | 535 | 80.00% | 417 | 76.28% | 71.70% | 73.92% |
| R | 527 | 87.38% | 692 | 83.58% | 57.37% | 68.04% |
| S | 424 | 86.09% | 605 | 33.23% | 92.56% | 48.91% |
| T | 540 | 98.69% | 701 | 67.49% | 31.10% | 42.58% |
| U | 502 | 83.43% | 508 | 66.67% | 5.51% | 10.18% |
| V | 473 | 100.00% | 774 | 87.10% | 13.95% | 24.05% |
| W | 408 | 100.00% | 338 | 99.70% | 98.82% | 99.26% |
| X | 564 | 82.01% | 781 | 36.11% | 77.72% | 49.31% |
| Y | 251 | 100.00% | 596 | 86.62% | 98.83% | 92.32% |

## Interpretation and limits

The candidate still has major class failures: validation G and M both have 0% F1, while test P, U and V have 10.06%, 10.18% and 24.05% F1. These source-conditioned failures need image/label-convention and pose-coverage review; the scores alone do not prove incorrect labels. The aggregate validation score hides those failures. This candidate should remain a diagnostic artifact rather than replace the application model.

Handedness is strongly confounded with source: training has 17,200 Left / 78,492 Right rows; validation has 1,238 Left / 10,043 Right; test has 15,688 Left / 47 Right. This test mainly measures transfer to left-handed examples from archive (3), not balanced performance across both hands. Source, handedness and other capture differences cannot be separated by this single experiment.

This is a source-transfer diagnostic, not verified signer-independent evaluation. No real or fictional signer IDs were assigned. Image matches reproduce the label, handedness and all 63 stored coordinates within 1e-6; multiple matching images remain recorded. The candidate excludes rows that cannot be linked by that evidence. This creates a recoverability-selected subset, not a representative random sample of either complete archive. Only one validation source and one test source are available. Transformed image overlap and unknown shared people across different archives are not ruled out. Some holdout rows and source examples were examined during earlier diagnostics; this is a development benchmark, not an untouched external test.

These scores cannot be directly compared with the historical 56.62% accuracy / 63.07% macro F1: the test rows and training distribution changed. The retained checkpoint is also contaminated for this new holdout, so it is not presented as an unseen-data baseline. This run establishes a reproducible source-separated baseline for future candidates; it does not establish that this candidate improves the deployed model.

The original dataset, labels and all existing .pth checkpoints were verified unchanged. The candidate was saved separately and was not deployed. J/Z dynamic recognition was not retrained.

[Protocol](protocol.json), [preparation and coverage](preparation.json), [all-row inclusion/exclusion ledger](row-ledger.csv), [results and confusion matrices](results.json), [training history](model.metrics.json), [validation predictions](validation-predictions.csv), [test predictions](test-predictions.csv).

[Independent output checks](verification.json) recompute the reported metrics from predictions and verify source/session ownership, row counts, all 50 epochs and unchanged original inputs.
