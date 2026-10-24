> Model cleanup: unused candidate weights were deleted at the owner's request on 24 October 2026. Historical metrics, predictions and source data remain available. The deployed static and J/Z models were retained. See [cleanup inventory](../../model-cleanup/20261024/report.md).

# Whole-model fine-tuning diagnostic

**Outcome: keep the retained application model.** All 50 epochs completed; validation selected epoch 28. Test accuracy fell from 56.62% to 51.44%, and macro F1 from 63.07% to 59.53%. The 10-epoch fine-tuned candidate was also better than this 50-epoch candidate overall. This run improved validation F1 to 97.73% while worsening test performance, reinforcing the validation-representativeness problem. No application checkpoint was replaced.

On the same 34,720 test rows, the candidate corrected 425 of the retained model's errors but made 2,222 previously correct rows wrong. These are sample counts, not independent people or recordings. Relative to the retained model, F1 declined particularly for U (-24.83 points), S (-19.24), O (-19.03), R (-16.41) and X (-16.02), while L/M/N improved.

All layers trainable, all 24 letters, all 1800 added M rows training-only. 50 epochs, LR 0.0001, seed 42, otherwise trainer defaults. Starts from the retained application checkpoint, not a previous fine-tuned candidate. Initial weights eligible at epoch 0; selected only by historical validation macro F1 then loss. Tests evaluated after selection. Cosine schedule uses the specified total epoch count.

Selected epoch: 28. Weight change L2 from starting checkpoint: 6.834462.

| Metric | Current model | From-scratch retraining | Previous 10-epoch fine-tuning | 50-epoch fine-tuning |
|---|---:|---:|---:|---:|
| Validation macro F1 | 97.33% | 97.57% | 97.61% | 97.73% |
| Test accuracy | 56.62% | 52.07% | 54.12% | 51.44% |
| Test macro F1 | 63.07% | 59.30% | 61.67% | 59.53% |
| M test F1 | 73.06% | 80.84% | 83.41% | 82.20% |
| N test F1 | 56.49% | 63.29% | 65.98% | 65.98% |
| Test macro F1 at confidence 0.8 | 66.27% | 64.81% | 65.88% | 64.54% |
| Accepted coverage at confidence 0.8 | 85.53% | 83.32% | 84.49% | 84.03% |
| Accepted-only accuracy at confidence 0.8 | 60.67% | 59.33% | 59.20% | 57.29% |

| Letter | Current F1 | Fine-tuned F1 | Change (pp) |
|---|---:|---:|---:|
| A | 81.48% | 79.73% | -1.76 |
| B | 96.73% | 95.56% | -1.17 |
| C | 43.88% | 45.88% | +2.00 |
| D | 33.02% | 28.75% | -4.27 |
| E | 77.41% | 74.87% | -2.54 |
| F | 95.44% | 96.39% | +0.95 |
| G | 21.72% | 21.50% | -0.22 |
| H | 36.32% | 27.56% | -8.76 |
| I | 96.57% | 96.46% | -0.11 |
| K | 45.70% | 44.78% | -0.92 |
| L | 36.83% | 54.19% | +17.36 |
| M | 73.06% | 82.20% | +9.14 |
| N | 56.49% | 65.98% | +9.49 |
| O | 72.18% | 53.15% | -19.03 |
| P | 18.76% | 17.84% | -0.92 |
| Q | 50.42% | 40.94% | -9.48 |
| R | 56.22% | 39.81% | -16.41 |
| S | 76.34% | 57.10% | -19.24 |
| T | 72.00% | 75.11% | +3.11 |
| U | 87.74% | 62.91% | -24.83 |
| V | 23.44% | 23.18% | -0.26 |
| W | 99.27% | 99.45% | +0.18 |
| X | 68.32% | 52.30% | -16.02 |
| Y | 94.42% | 93.15% | -1.26 |

Diagnostic only. Historical test is development-known; signer independence unverified. One seed and one predefined configuration, not a controlled multi-seed causal comparison. No new signer-independent or live-camera evidence.

Validation: 13,791 rows; test: 34,720 rows; training: 111,023 rows. Rejected known signs count as false negatives in thresholded F1. Accepted-only accuracy must be read with coverage. No thresholds were fitted, and no evaluation rows were used for gradient updates.

Existing data and application checkpoints were verified unchanged. No application model was replaced. Full metrics, confusions, hashes and provenance are in [comparison.json](comparison.json).

## What to try next

1. **Audit the difficult data before adding epochs.** Review original evidence for high-count confusions, including K→V (3,323 rows), Q→P (2,647), G→H (2,566), and O→D (1,693) in this candidate. Verify the intended alphabet convention, labels, detector quality, handedness and viewpoint by source. Preserve originals and record decisions; do not relabel samples merely to agree with predictions. The earlier source analysis localized much of the poor performance to a large unresolved import group.
2. **Make validation reflect the failure cases and collect a fresh test.** Use real, possibly anonymous person identities when available; source-aware diagnostics can help while identities remain unknown, but are not signer-independent evaluation. The repeated pattern of higher validation F1 and lower test F1 means that optimizing this validation score alone is insufficient. Keep the historical test as development evidence, and avoid choosing new settings based on its repeated results.
3. **Try fine-tuning that explicitly preserves the starting model's behavior.** The next model experiment should compare current ordinary fine-tuning against teacher-output regularization on existing all-letter training data, or a penalty for moving far from the starting weights. These directly address changes to previously correct predictions, unlike simply increasing epochs. They may also preserve old mistakes; improvements are not guaranteed. Use representative validation and per-letter regression limits.
4. **Limit what can change.** Compare freezing earlier feature layers and updating only the classifier/top layers, possibly with a lower learning rate, against the retained baseline. Keep all-letter examples in training and inspect class/source sampling and inverse-class weighting; more frames from a narrow source should not dominate. Specify a small set of trials in advance and check consistency across seeds rather than searching until the historical test improves.
5. **Validate the deployed recognition path.** Offline landmark F1 does not measure detection failures, idle/unknown gestures, motion routing, duplicate emissions or latency. Use continuous-stream replay and actual browser/Python inference parity before promoting any candidate. Threshold calibration may improve the rejection tradeoff, but cannot repair incorrect labels or missing pose coverage.

The recommended order is data/validation review first, then a preservation-constrained fine-tuning experiment. A larger network and more epochs are not established remedies from the measured evidence. None of the proposed next experiments has been run as part of this trial.

## Verification

Eight prediction tables were checked against sample counts, row IDs, accuracies and full confusion matrices. Recomputed retained/from-scratch results match the prior benchmark, and the 10-epoch predictions reproduce the earlier trial. Input/checkpoint hashes confirm that the source CSV, starting model, prior candidates and other application models remain unchanged. The selected model is saved separately as `model.pth` in this directory.
