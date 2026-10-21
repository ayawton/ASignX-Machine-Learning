> **Original-test evaluation:** This report uses the pre-deletion test set (36,973 samples). For the comparison on the current cleaned set (34,720 samples), see [the current comparison](../../current-model-comparison/report.md).

# Pre-deletion dataset training

Trained 50 epochs on 159,987 original samples; selected epoch 43 by validation only.

| Metric | Existing model | Retrained model |
|---|---:|---:|
| Validation macro F1 | 97.33% | 97.72% |
| Test macro F1 | 59.49% | 54.13% |
| Test accuracy | 53.19% | 46.56% |
| Test weighted F1 | 56.16% | 48.58% |
| M test F1 | 36.73% | 33.17% |
| N test F1 | 22.81% | 22.86% |

Original partitions: training 109,223; validation 13,791; test 36,973. The archived pre-deletion CSV was used directly. The cleaned main CSV and existing application models are unchanged.

Development-known session split; signer independence unverified. Offline, unthresholded static landmark classification. Original disputed signs remain in this historical test.
