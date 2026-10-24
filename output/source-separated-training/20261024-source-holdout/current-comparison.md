> Model cleanup: unused candidate weights were deleted at the owner's request on 24 October 2026. Historical metrics, predictions and source data remain available. The deployed static and J/Z models were retained. See [cleanup inventory](../../model-cleanup/20261024/report.md).

# Current model versus source-separated candidate

Both saved checkpoints were scored on exactly the same candidate validation/test rows, using each model's recorded handedness preprocessing. No retraining, threshold fitting or checkpoint selection was performed for this comparison.

| Partition | Metric | Current deployed model | New candidate | Candidate change (pp) |
|---|---|---:|---:|---:|
| Validation (11,281) | accuracy | 99.82% | 84.84% | -14.98 |
| Validation (11,281) | macro_f1 | 99.80% | 83.08% | -16.73 |
| Test (15,735) | accuracy | 97.88% | 62.84% | -35.04 |
| Test (15,735) | macro_f1 | 97.73% | 59.79% | -37.94 |

The current checkpoint leads on both the full test and the shared historical-test subset. Keep it deployed; this candidate does not demonstrate an improvement. Source/person overlap still prevents a signer-independent conclusion.

## Exposure audit

The full comparison includes a substantial exposure advantage for the current model: it trained on part of these candidate holdouts. The audit uses its original dataset, verified against the dataset SHA-256 embedded in its checkpoint, and its actual saved split indices. It also checks exact canonical geometry against all original training rows.

| Candidate partition | Current original train rows | Current original validation rows | Current original test rows | Training-geometry overlap |
|---|---:|---:|---:|---:|
| validation | 7,883 | 1,879 | 1,519 | 7,883 |
| test | 11,033 | 2,212 | 2,490 | 11,033 |

## Shared historical-test subset

This secondary view keeps only rows assigned to test by the current model's original manifest, excluding any canonical geometry found in its training or validation. These rows also belong to the candidate holdout. It removes direct fitted-row and validation-selection exposure, but the current model still trained on other examples from these source families and potentially the same people. The historical test has also been inspected in prior development. This is a diagnostic comparison, not proof of independent-person generalization.

| Candidate partition / shared subset | Metric | Current | Candidate | Candidate change (pp) |
|---|---|---:|---:|---:|---:|
| validation (1,519; 24 labels) | accuracy | 99.87% | 87.76% | -12.11 |
| validation (1,519; 24 labels) | macro_f1 | 99.84% | 84.76% | -15.08 |
| test (2,490; 24 labels) | accuracy | 97.55% | 61.16% | -36.39 |
| test (2,490; 24 labels) | macro_f1 | 97.51% | 56.84% | -40.67 |

Macro F1 always uses the 24 model labels. Read shared-subset label coverage before comparing it with full-partition scores.

## All-letter full test comparison

| Letter | Test rows | Current F1 | Candidate F1 | Candidate change (pp) |
|---|---:|---:|---:|---:|
| A | 425 | 94.79% | 84.21% | -10.58 |
| B | 267 | 98.31% | 43.71% | -54.60 |
| C | 823 | 99.27% | 30.99% | -68.28 |
| D | 672 | 97.84% | 60.68% | -37.16 |
| E | 555 | 97.59% | 51.79% | -45.81 |
| F | 633 | 99.06% | 95.35% | -3.70 |
| G | 1,114 | 99.06% | 82.11% | -16.95 |
| H | 1,145 | 99.30% | 85.50% | -13.81 |
| I | 762 | 98.68% | 82.61% | -16.07 |
| K | 840 | 98.50% | 32.98% | -65.52 |
| L | 716 | 99.30% | 97.61% | -1.69 |
| M | 637 | 96.64% | 72.95% | -23.69 |
| N | 673 | 95.95% | 32.97% | -62.99 |
| O | 599 | 97.38% | 62.83% | -34.55 |
| P | 462 | 95.99% | 10.06% | -85.93 |
| Q | 417 | 97.88% | 73.92% | -23.96 |
| R | 692 | 96.14% | 68.04% | -28.10 |
| S | 605 | 95.21% | 48.91% | -46.30 |
| T | 701 | 95.61% | 42.58% | -53.03 |
| U | 508 | 96.79% | 10.18% | -86.60 |
| V | 774 | 97.95% | 24.05% | -73.90 |
| W | 338 | 99.41% | 99.26% | -0.15 |
| X | 781 | 99.17% | 49.31% | -49.86 |
| Y | 596 | 99.66% | 92.32% | -7.34 |

On the full test, the candidate fixes 55 current-model errors and breaks 5,569 previously correct predictions. The source-transfer benchmark is nearly all Left-handed in test and does not establish balanced performance across both hands.

Dataset and both checkpoints were verified unchanged. This comparison does not promote either checkpoint.

[Machine-readable scores, confusion matrices and exposure counts](current-comparison.json), [combined validation predictions](current-comparison-validation-predictions.csv), [combined test predictions](current-comparison-test-predictions.csv), [comparison script](compare_current.py).
