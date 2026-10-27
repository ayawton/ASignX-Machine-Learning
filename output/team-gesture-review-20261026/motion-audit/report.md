# Sequence motion audit

Model: `output\pilot-training\20261026T114120Z-hello-retake-9007e3\sequence_left_pilot.pth`; split: **validation**; threshold: 80%.

Counts show predictions accepted as the recording label. Altered sequences are diagnostic probes, not labeled test accuracy.

| Label | Input | Accepted / recordings | Mean score for recording label |
|---|---|---:|---:|
| HELLO | original | 1 / 1 | 99.7% |
| HELLO | frozen_first | 1 / 1 | 99.6% |
| HELLO | frozen_middle | 1 / 1 | 99.8% |
| HELLO | reversed | 1 / 1 | 99.7% |
| HELLO | first_half_stretched | 1 / 1 | 99.6% |
| HELLO | shuffled | 1 / 1 | 99.7% |
| J | original | 1 / 1 | 98.6% |
| J | frozen_first | 1 / 1 | 89.4% |
| J | frozen_middle | 0 / 1 | 8.9% |
| J | reversed | 1 / 1 | 98.5% |
| J | first_half_stretched | 1 / 1 | 95.9% |
| J | shuffled | 1 / 1 | 98.9% |
| Z | original | 2 / 2 | 100.0% |
| Z | frozen_first | 1 / 2 | 68.3% |
| Z | frozen_middle | 1 / 2 | 79.3% |
| Z | reversed | 2 / 2 | 100.0% |
| Z | first_half_stretched | 2 / 2 | 100.0% |
| Z | shuffled | 2 / 2 | 100.0% |

Limitations:
- Raw classifier outputs: bypasses live motion gating and visibility rejection.
- Held-out recordings follow saved session splits; they are not necessarily unseen signers.
- Reversed or truncated examples are not automatically invalid ASL gestures.
- Confidence is a model score, not a calibrated probability of ASL correctness.
- For hand-only trajectory-anchored formats, reversal does not recompute the original raw anchor.
