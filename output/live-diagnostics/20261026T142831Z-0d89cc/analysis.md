# Live pilot findings

Source: summary.json and frames.jsonl from this session. No model or threshold
changes were applied to the source session. User-selected blocks are intentions,
not verified annotations of individual ASL performances.

| Block | Observed outputs and decisions |
| --- | --- |
| J | 7 classifier evaluations: 5 J emissions; 2 below-threshold results (HELLO .598 and J .497) |
| Z | 6 Z emissions; 1 short-capture rejection |
| HELLO | Only 2 classifier evaluations: Z .988 and HELLO .879; no tracking rejection |
| REST | No sign emissions; 2 insufficient-motion rejections |

These counts are not attempt accuracy: an attempt counter or verified raw video
was not captured, and one performance can be split or missed by the motion gate.

## Timing and tracking

Overall observed pipeline throughput: 15.48 FPS. HELLO block: 14.89 FPS. Median
detector time: 32.48 ms overall; p95 41.91 ms. No logged capture interval exceeded
200 ms. All frames reported visible shoulders and face references. Whole-block
HELLO hand visibility was 80.2%, including pauses; both captured HELLO spans had
visible hands on every processed frame. Timing does not prove absence of blur.

## Primary finding

The start threshold was .045. In HELLO, only 5 of 187 processed frames exceeded
it; velocity p95 was .0348. The router remained static for 170 frames. It captured
only spans 833-840 and 916-924, lasting .46 and .54 seconds before finalization.
This supports missed or late capture as a leading issue in this session.

The confident Z output in the HELLO block also demonstrates a classification or
input-segmentation mismatch. The logged timing/visibility flags contain no raw
landmark trajectories, so they cannot establish which caused that particular
misclassification or verify signing correctness.

## Next controlled test

Run ASignX_DiagnosePilot_Sensitive.bat, which changes only the motion-start
threshold to .025 (and the existing proportional exit threshold). Retain the same
checkpoint, confidence threshold, tracking checks and sustained-motion guard.
Repeat the J/Z/HELLO and REST sections using keys 7/8/9/0, then exit with Q.
This is a development comparison, not a calibrated release threshold. Examine
whether HELLO reaches classification more consistently and whether ordinary
movement creates false emissions. Do not lower confidence to disguise confusion.
