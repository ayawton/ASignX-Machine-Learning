# Recognition reevaluation

Fresh inference on the fixed historical validation/test rows. Inputs and checkpoints are unchanged. No training or threshold fitting.

| Metric | Retained | Candidate |
|---|---:|---:|
| Validation macro F1 | 97.33% | 97.85% |
| Test accuracy | 53.19% | 48.01% |
| Test macro F1 | 59.49% | 56.49% |
| Test macro F1 at threshold 0.8 | 63.06% | 60.98% |
| Accepted test coverage at threshold 0.8 | 85.79% | 85.13% |
| Accepted-only test accuracy at threshold 0.8 | 56.81% | 53.60% |
| M test F1 | 36.73% | 46.72% |
| N test F1 | 22.81% | 24.85% |

Rejected known signs count as false negatives in thresholded F1. Accepted-only accuracy excludes rejected samples and must be read alongside coverage. Static validation contains 13,791 samples; test contains 36,973.

Dynamic J/Z evaluation: 100% test macro F1 on only two clips (one per class). This is insufficient to establish reliability or movement understanding.

These are offline landmark scores, not a live camera or continuous-stream benchmark. Historical partitions are development-known and not verified signer-independent. Repeated evaluation does not create new independent evidence.
