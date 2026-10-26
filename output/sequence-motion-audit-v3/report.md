# Sequence motion audit

Model: `ml/models/sequence_body_v3.pth`; split: **test**; threshold: 80%.

Counts show predictions accepted as the recording label. Altered sequences are diagnostic probes, not labeled test accuracy.

| Label | Input | Accepted / recordings | Mean score for recording label |
|---|---|---:|---:|
| Hello | original | 7 / 7 | 99.3% |
| Hello | frozen_first | 7 / 7 | 99.8% |
| Hello | frozen_middle | 4 / 7 | 57.3% |
| Hello | reversed | 7 / 7 | 99.3% |
| Hello | first_half_stretched | 7 / 7 | 99.8% |
| Hello | shuffled | 7 / 7 | 99.3% |
| J | original | 2 / 3 | 89.5% |
| J | frozen_first | 3 / 3 | 99.7% |
| J | frozen_middle | 3 / 3 | 99.8% |
| J | reversed | 2 / 3 | 89.4% |
| J | first_half_stretched | 3 / 3 | 99.7% |
| J | shuffled | 3 / 3 | 97.8% |

Limitations:
- Raw classifier outputs: bypasses live motion gating and visibility rejection.
- Held-out recordings follow saved session splits; they are not necessarily unseen signers.
- Reversed or truncated examples are not automatically invalid ASL gestures.
- Confidence is a model score, not a calibrated probability of ASL correctness.
- For hand-only trajectory-anchored formats, reversal does not recompute the original raw anchor.
