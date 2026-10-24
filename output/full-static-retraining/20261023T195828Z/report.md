> Model cleanup: unused candidate weights were deleted at the owner's request on 24 October 2026. Historical metrics, predictions and source data remain available. The deployed static and J/Z models were retained. See [cleanup inventory](../../model-cleanup/20261024/report.md).

# Full static alphabet retraining

**Result:** Predictions changed, but full retraining did not outperform the current application model overall. Compared with the previous 1,400-M candidate, test accuracy improved by 2.75 percentage points and macro F1 by 0.25 points, while M F1 decreased by 1.44 points. Compared with the current application, test accuracy is 4.54 points lower and macro F1 is 3.77 points lower; M F1 is 7.78 points higher. Retain the current application model.

Selected checkpoint: epoch 47 of 50. The retrained model is saved as `model.pth` in this directory.

Retrained every layer of the 24-letter static classifier from scratch for 50 epochs, with all 1,800 new M samples included in training. Training: 111,023; validation: 13,791; test: 34,720. Selected by historical validation macro F1, then loss.

| Metric | Current application | Previous 1,400-M candidate | Full retraining |
|---|---:|---:|---:|
| Validation macro F1 | 97.33% | 97.70% | 97.57% |
| Test accuracy | 56.62% | 49.32% | 52.07% |
| Test macro F1 | 63.07% | 59.05% | 59.30% |
| M test F1 | 73.06% | 82.28% | 80.84% |
| N test F1 | 56.49% | 65.81% | 63.29% |
| Test macro F1 at confidence 0.8 | 66.27% | 63.08% | 64.81% |
| Coverage at confidence 0.8 | 85.53% | 85.32% | 83.32% |
| Accepted-only accuracy at confidence 0.8 | 60.67% | 55.66% | 59.33% |

Rejected known signs count as false negatives in thresholded F1. Accepted-only accuracy must be read alongside coverage. The confidence threshold was not fitted.

| Letter | Current application F1 | Previous candidate F1 | Full retraining F1 | Change vs application (pp) |
|---|---:|---:|---:|---:|
| A | 81.48% | 81.83% | 79.93% | -1.55 |
| B | 96.73% | 97.73% | 97.93% | +1.20 |
| C | 43.88% | 36.75% | 45.16% | +1.29 |
| D | 33.02% | 28.36% | 32.10% | -0.92 |
| E | 77.41% | 79.68% | 72.07% | -5.34 |
| F | 95.44% | 90.74% | 96.75% | +1.31 |
| G | 21.72% | 21.51% | 21.45% | -0.26 |
| H | 36.32% | 35.62% | 29.45% | -6.86 |
| I | 96.57% | 96.52% | 96.59% | +0.02 |
| K | 45.70% | 45.08% | 46.27% | +0.57 |
| L | 36.83% | 42.84% | 48.54% | +11.71 |
| M | 73.06% | 82.28% | 80.84% | +7.78 |
| N | 56.49% | 65.81% | 63.29% | +6.80 |
| O | 72.18% | 43.92% | 57.82% | -14.36 |
| P | 18.76% | 18.59% | 19.56% | +0.79 |
| Q | 50.42% | 41.94% | 48.81% | -1.60 |
| R | 56.22% | 27.60% | 35.36% | -20.87 |
| S | 76.34% | 60.98% | 52.88% | -23.46 |
| T | 72.00% | 79.48% | 79.25% | +7.25 |
| U | 87.74% | 91.10% | 61.82% | -25.92 |
| V | 23.44% | 23.33% | 23.76% | +0.32 |
| W | 99.27% | 99.45% | 99.27% | -0.00 |
| X | 68.32% | 40.54% | 44.69% | -23.62 |
| Y | 94.42% | 85.40% | 89.56% | -4.86 |

Single seed; historical test is development-known and signer independence unverified. The previous 400 new-M holdout rows are now training data and cannot be reported as held-out evaluation. Dynamic J/Z data and model are not part of this static retraining.

The previous matched experiment already retrained the full static model; this run adds the remaining 400 new M samples to training. All comparisons above use the same 34,720 historical test rows. Existing application checkpoints and source dataset were verified unchanged. These offline landmark scores do not measure live-camera performance.

See [comparison.json](comparison.json) for metrics, confusion matrices, model settings, and hashes; [preparation.json](preparation.json) records the training protocol.
