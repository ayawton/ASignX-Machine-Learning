> Model cleanup: unused candidate weights were deleted at the owner's request on 24 October 2026. Historical metrics, predictions and source data remain available. The deployed static and J/Z models were retained. See [cleanup inventory](../../model-cleanup/20261024/report.md).

# Added M samples: matched reevaluation

**M improved in this matched run, but this candidate is not an overall replacement for the retained application model.** M test F1 increased by 14.14 percentage points (68.14% to 82.28%); N F1 increased by 9.61 points. Overall macro F1 increased by 1.27 points, while accuracy decreased by 0.25 points. The retained application model still has higher overall macro F1 (63.07% versus 59.05%) and accuracy (56.62% versus 49.32%). Keep the current application checkpoint.

The M gain on the historical test is concentrated in right-hand examples: right M F1 rose from 67.92% to 87.97%, while left M F1 was essentially unchanged (68.61% to 68.31%). Left/right canonicalization allows left-hand training data to influence right-hand predictions. M-to-N errors fell from 489 to 296. The new left-hand session holdout improved from 247/400 correct to 400/400, but shared signer/setup similarity remains possible. At the fixed 0.8 confidence threshold, overall macro F1 slightly declined (63.29% to 63.08%).

M samples increased from 3,388 to 5,188: 1,800 new left-hand samples in 9 sessions. Seven sessions (1,400 samples) were added to training; two sessions (400 samples) were held out. Existing rows match the pre-addition backup exactly as loaded float32 samples.

| Metric | Retained app model | Before additions | After additions |
|---|---:|---:|---:|
| Historical validation macro F1 | 97.33% | 97.72% | 97.70% |
| Historical test accuracy | 56.62% | 49.57% | 49.32% |
| Historical test macro F1 | 63.07% | 57.77% | 59.05% |
| Historical M precision | 94.79% | 97.18% | 97.32% |
| Historical M recall | 59.43% | 52.47% | 71.26% |
| Historical M F1 | 73.06% | 68.14% | 82.28% |
| Historical N F1 | 56.49% | 56.20% | 65.81% |
| New M session holdout recall / accuracy | 77.00% | 61.75% | 100.00% |

Both fresh models use identical 50-epoch settings and seed 42, selected only by the unchanged historical validation set. Historical test: 34,720 rows; validation: 13,791 rows. No test samples were added to training. The new holdout contains only M, so its recall/accuracy does not measure false-positive behavior on other signs.

One seed. Historical test is development-known; signer independence is unknown. New session holdout may share a signer or recording setup; no source images available for pose-label verification.

This measures the effect of adding seven of the nine sessions in one matched run, not a model trained on all 1,800 new samples. Scores are unthresholded offline landmark classification; live-camera performance is not measured. Existing data and application checkpoints were verified unchanged.

See [comparison.json](comparison.json) for all classes, confusion matrices, hand breakdowns, hashes, and configurations.

## Detailed comparison

All changes below compare the matched after model with the matched before model. Values are percentage points.

| Letter | Before test F1 | After test F1 | Change |
|---|---:|---:|---:|
| A | 79.45% | 81.83% | +2.38 |
| B | 97.53% | 97.73% | +0.20 |
| C | 45.37% | 36.75% | -8.61 |
| D | 26.49% | 28.36% | +1.87 |
| E | 67.18% | 79.68% | +12.51 |
| F | 97.24% | 90.74% | -6.50 |
| G | 21.64% | 21.51% | -0.13 |
| H | 26.18% | 35.62% | +9.44 |
| I | 96.49% | 96.52% | +0.03 |
| K | 45.63% | 45.08% | -0.56 |
| L | 57.50% | 42.84% | -14.66 |
| M | 68.14% | 82.28% | +14.14 |
| N | 56.20% | 65.81% | +9.61 |
| O | 52.69% | 43.92% | -8.77 |
| P | 17.79% | 18.59% | +0.80 |
| Q | 37.45% | 41.94% | +4.49 |
| R | 45.52% | 27.60% | -17.91 |
| S | 60.46% | 60.98% | +0.52 |
| T | 82.82% | 79.48% | -3.34 |
| U | 49.16% | 91.10% | +41.94 |
| V | 23.07% | 23.33% | +0.27 |
| W | 99.63% | 99.45% | -0.18 |
| X | 43.81% | 40.54% | -3.27 |
| Y | 89.17% | 85.40% | -3.77 |

| M historical test breakdown | Retained app model | Before additions | After additions |
|---|---:|---:|---:|
| Left M precision | 88.39% | 94.00% | 92.19% |
| Left M recall | 54.25% | 54.02% | 54.25% |
| Left M f1 | 67.24% | 68.61% | 68.31% |
| Left M support | 435 | 435 | 435 |
| Right M precision | 97.65% | 98.79% | 99.07% |
| Right M recall | 61.82% | 51.75% | 79.11% |
| Right M f1 | 75.71% | 67.92% | 87.97% |
| Right M support | 943 | 943 | 943 |
| M predicted as N | 435 | 489 | 296 |
| N predicted as M | 3 | 1 | 1 |
| All non-M samples predicted as M | 45 | 21 | 27 |

| Historical test at confidence 0.8 | Retained app model | Before additions | After additions |
|---|---:|---:|---:|
| Macro F1 | 66.27% | 63.29% | 63.08% |
| Coverage | 85.53% | 84.39% | 85.32% |
| Accepted-only accuracy | 60.67% | 55.70% | 55.66% |

Rejected known signs count as false negatives in thresholded F1. Accepted-only accuracy excludes rejected rows and must be read with coverage. The threshold was fixed in advance; it was not fitted to these results.
