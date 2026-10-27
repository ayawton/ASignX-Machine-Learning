# Reviewed evaluation labels

**Annotation clarification pending:** after this review, the user highlighted
that a HELLO stopped early can simply be a short valid HELLO. The earlier
collection instruction was ambiguous. The decisions and recomputed counts below
are retained as historical evidence, but affected negative examples require
reconsideration under the clarified rule before these counts support a quality
claim. No decisions, source files or model weights were automatically changed.

All 15 original BACKGROUND examples were reviewed by the user without model
scores shown in the reviewer: 11 were retained as BACKGROUND and 4 were judged
valid HELLO forms. No examples were marked UNSURE. Reviewed source hashes match.

The original files and labels remain unchanged. A separate annotation manifest
records the user's judgments; none of these examples was added to training and
no confidence threshold or model was changed.

## Recomputed prepared-clip decisions

Combining those judgments with the original 5 HELLO labels gives 9 HELLO and
11 BACKGROUND examples. The same training-v2 model, at confidence .8, produces:

| Reviewed category | Correct outcome | Remaining issue |
| --- | ---: | --- |
| HELLO | 8 accepted / 9 | 1 missed (raw Z at .712, no sign emitted) |
| BACKGROUND | 10 rejected / 11 | 1 incorrectly accepted as HELLO at .960 |

The remaining files are BACKGROUND_575eca42eab5482bbd8b99bc64b59939.npz (reviewed
HELLO) and BACKGROUND_bdaf7dc01e844a0693e6b579edaba183.npz (reviewed BACKGROUND).
Prediction details and fingerprints are in reviewed_evaluation.json.

These are user-reviewed landmark replays from one signer and already inspected
development data. They are not an independent final-test accuracy estimate.
Reviewing only selected original negatives does not establish complete annotation
quality for all classes. The judgments do not include original video or an
independent ASL assessment. Live segmentation and guards are outside these
prepared-clip decisions.

Before future training, audit training-set negatives against the same salute /
greeting-wave definition. Do not copy the reviewed evaluation examples into
training or lower confidence just to fit the missed example.
