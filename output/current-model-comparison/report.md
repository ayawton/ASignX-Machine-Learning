# Current model comparison: cleaned evaluation set

Both saved models were evaluated on exactly the same reviewed samples. No retraining, checkpoint replacement, threshold tuning, or additional data deletion occurred.

| Metric | Existing model | Retrained model |
|---|---:|---:|
| Validation macro F1 | 97.33% | 97.72% |
| Test macro F1 | 63.07% | 57.77% |
| Test accuracy | 56.62% | 49.57% |
| M test F1 | 73.06% | 68.14% |
| N test F1 | 56.49% | 56.20% |

Evaluation: 34,720 cleaned test samples and 13,791 validation samples. Dataset SHA-256: 4a160bec73556408c12f9457577c765c89e9f8cf09eae3d658fe39d3064f58cd

The retrained checkpoint was trained from the archived pre-deletion snapshot. Its training metadata is preserved. Evaluation uses the current cleaned snapshot; the two dataset identities are deliberately recorded separately.

The 54.13% retrained macro F1 and 59.49% existing-model macro F1 remain valid results for the original 36,973-sample test set. They are a separate historical evaluation, not scores on this cleaned set.

Unthresholded offline static landmark classification. Development-known session split; signer independence unverified. Test cleanup changes evaluation composition, not learned weights.
