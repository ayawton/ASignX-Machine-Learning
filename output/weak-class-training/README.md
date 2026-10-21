# Weak-class training candidate

This is a diagnostic retraining experiment, separate from the active dataset and models.

- `combined.csv`: original 159,987 rows plus 4,118 newly extracted image samples.
- `preparation.json`: preparation counts and overlap policy.
- `overlap-review.csv`: nearest original sample for every new sample, using canonical landmark L2 distance across all original partitions. No new samples fell within the 0.015 exclusion threshold. This does not rule out transformed duplicates, shared signers or shared original sources.
- `model.splits.json`: frozen original validation and test indices; all new samples added only to training. Historical grouping is not signer-independent.
- `model.pth`: best candidate checkpoint, chosen by validation macro F1, then loss.
- `model.metrics.json`: training history and selected validation result. Test evaluation is disabled during training.
- `comparison.json`: completed-run comparison with the incumbent on identical original test rows, and unchanged-input hash checks.
- `candidate-test-predictions.csv` and `incumbent-test-predictions.csv`: comparable row-level test results.

Training uses 50 epochs, batch size 64, seed 42, the canonical residual MLP architecture, and existing trainer defaults. It starts from scratch. No production checkpoint or launcher is replaced.

The historical test has already informed development. New signer identities are unknown. This experiment cannot establish independent new-signer generalization or isolate the added data's effect from retraining variation. All original labels are retained without model-driven relabeling.

Once training completes, try the candidate explicitly:

```powershell
py -3.11 main.py --sign-model output/weak-class-training/model.pth --exposure auto
```

Source-image traceability for the new rows is in `output/datasets/weak-classes/landmark-provenance.csv`, where `csv_row` refers to the standalone extracted CSV, not the combined CSV. New rows are appended in that CSV's order. Their combined CSV line number equals their standalone CSV line number plus 159,987.

The orchestration script is `tmp/train_weak_candidate.py`; the run log is `output/weak-class-training-run.log`. Do not rerun that script into the same output directory: it refuses to overwrite candidate data or an existing checkpoint.

## Completed run

50 epochs completed; validation selected epoch 36. Validation macro F1 was 97.85%.

| Metric | Incumbent | Candidate |
|---|---:|---:|
| Test accuracy | 53.19% | 48.01% |
| Test macro F1 | 59.49% | 56.49% |
| M test F1 | 36.73% | 46.72% |
| N test F1 | 22.81% | 24.85% |

Decision: retain the incumbent. M/N improved on the historical test, but overall accuracy and macro F1 regressed. This is not a successful overall replacement. A single from-scratch run does not isolate the added dataset as the cause of the regression. Checksum verification confirmed the original CSV and original static/dynamic checkpoints were unchanged.
