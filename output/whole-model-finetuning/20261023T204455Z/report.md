> Model cleanup: unused candidate weights were deleted at the owner's request on 24 October 2026. Historical metrics, predictions and source data remain available. The deployed static and J/Z models were retained. See [cleanup inventory](../../model-cleanup/20261024/report.md).

# Whole-model fine-tuning diagnostic

**Outcome:** Whole-model fine-tuning improved over the from-scratch candidate but did not beat the current application overall. Keep the current application model. Test accuracy was 54.12% versus 56.62%, and macro F1 was 61.67% versus 63.07%. M/N F1 improved to 83.41%/65.98%, while other letters regressed; the per-letter table below exposes those tradeoffs. Higher validation F1 did not translate to a better historical test result.

All layers trainable, all 24 letters, all 1800 added M rows training-only. Ten epochs, LR 0.0001, seed 42, otherwise trainer defaults. Initial weights eligible at epoch 0; selected only by historical validation macro F1 then loss. Tests evaluated after selection.

Selected epoch: 8. Weight change L2 from starting checkpoint: 3.348118.

| Metric | Current model | From-scratch retraining | Fine-tuning |
|---|---:|---:|---:|
| Validation macro F1 | 97.33% | 97.57% | 97.61% |
| Test accuracy | 56.62% | 52.07% | 54.12% |
| Test macro F1 | 63.07% | 59.30% | 61.67% |
| M test F1 | 73.06% | 80.84% | 83.41% |
| N test F1 | 56.49% | 63.29% | 65.98% |
| Test macro F1 at confidence 0.8 | 66.27% | 64.81% | 65.88% |
| Accepted coverage at confidence 0.8 | 85.53% | 83.32% | 84.49% |
| Accepted-only accuracy at confidence 0.8 | 60.67% | 59.33% | 59.20% |

| Letter | Current F1 | Fine-tuned F1 | Change (pp) |
|---|---:|---:|---:|
| A | 81.48% | 80.07% | -1.41 |
| B | 96.73% | 96.93% | +0.20 |
| C | 43.88% | 46.51% | +2.63 |
| D | 33.02% | 32.25% | -0.77 |
| E | 77.41% | 78.70% | +1.29 |
| F | 95.44% | 96.15% | +0.71 |
| G | 21.72% | 21.58% | -0.14 |
| H | 36.32% | 28.70% | -7.62 |
| I | 96.57% | 96.53% | -0.04 |
| K | 45.70% | 44.96% | -0.74 |
| L | 36.83% | 53.26% | +16.43 |
| M | 73.06% | 83.41% | +10.35 |
| N | 56.49% | 65.98% | +9.49 |
| O | 72.18% | 65.53% | -6.64 |
| P | 18.76% | 17.64% | -1.12 |
| Q | 50.42% | 42.87% | -7.55 |
| R | 56.22% | 41.94% | -14.28 |
| S | 76.34% | 64.13% | -12.21 |
| T | 72.00% | 73.59% | +1.59 |
| U | 87.74% | 73.36% | -14.38 |
| V | 23.44% | 22.89% | -0.55 |
| W | 99.27% | 99.63% | +0.36 |
| X | 68.32% | 59.27% | -9.05 |
| Y | 94.42% | 94.28% | -0.14 |

Diagnostic only. Historical test is development-known; signer independence unverified. One seed and one predefined configuration, not a controlled multi-seed causal comparison. No new signer-independent or live-camera evidence.

Validation: 13,791 rows; test: 34,720 rows; training: 111,023 rows. Rejected known signs count as false negatives in thresholded F1. Accepted-only accuracy must be read with coverage. No thresholds were fitted, and no evaluation rows were used for gradient updates.

Existing data and application checkpoints were verified unchanged. No application model was replaced. Full metrics, confusions, hashes and provenance are in [comparison.json](comparison.json).
