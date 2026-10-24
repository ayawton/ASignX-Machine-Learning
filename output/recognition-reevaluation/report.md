> Storage cleanup completed: [deleted duplicate datasets and temporary artifacts, with verified restoration recipes](../storage-audit/20261024/completed-cleanup.md). Active data, models and recognition applications were preserved.

> Model cleanup completed: [retained production models and deleted unused artifacts](../model-cleanup/20261024/report.md). Historical experiment weights referenced below were removed; their evaluation records remain.

> Latest direct comparison: [current model versus source-separated candidate, including shared held-out rows and all-letter F1](../source-separated-training/20261024-source-holdout/current-comparison.md).

> Latest source-separated experiment: [50-epoch candidate and full-letter report](../source-separated-training/20261024-source-holdout/report.md). This uses a new benchmark; historical scores below are not directly comparable.

> Recovered source grouping: [applied constraints and independence limitations](../source-recovery/grouping.md).

> Source investigation: [recovered archives, image evidence and source-specific failures](../source-recovery/report.md).

> Latest 50-epoch fine-tuning: [results and recommended next steps](../whole-model-finetuning/20261023T205534Z/report.md).

> Latest whole-model fine-tuning trial: [comparison with the retained model](../whole-model-finetuning/20261023T204455Z/report.md).

> Live-state fixes and signer-benchmark readiness: [implementation and verification](../system-diagnosis/fixes.md).

> Whole-system diagnosis: [24 October 2026 findings and priorities](../system-diagnosis/report.md).

> Latest full static retraining: [results with all 1,800 new M samples](../full-static-retraining/20261023T195828Z/report.md). The report below describes earlier experiments.

> New M-sample experiment: [matched before/after reevaluation](../added-M-reevaluation/20261023T190744Z/report.md). The report below describes the earlier dataset and models.

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
