# HELLO / BACKGROUND movement-learning experiment

## Question and control

Does exposing signed frame-to-frame movement help distinguish complete HELLO
from a held or repositioning hand? The baseline is the four-class checkpoint at
../20261026T145427Z-background-768db7/sequence_left_background_pilot.pth.

This candidate changes one model option: use_motion_deltas=True. It uses the
same 36 copied recordings, person metadata, seed, positive and background
partitions, training settings and live confidence threshold (.8). No threshold
was fitted, no synthetic negatives were labeled as human performances, and the
saved test set was not evaluated. The existing checkpoints were hash-checked
and preserved.

## Why investigate movement

The baseline clip review is saved in
../20261026T145427Z-background-768db7/hello-background-motion-review.json.
For the HELLO validation example, the baseline HELLO score was .670 on the
original clip, .746 on a frozen first frame, and .669 on the reversed clip.
The original clip contains wrist movement: its visible trajectory totals about
.491 shoulder-width units, compared with .167 for the held-out background.
This supports testing movement representation. It does not verify signing
correctness or prove that total movement alone identifies HELLO.

## Implementation

ml/motion_features.py exposes signed changes of each coordinate. Derivatives
are scaled to normalized gesture progress because clips are resampled, not to
physical seconds. Hand changes across missing detections are masked. Body/face
coordinate changes require visibility at both observations; visibility bits
themselves never count as displacement.

The 180 original features remain the external input. The model internally
appends 180 movement features. Training enables this only with --motion-deltas.
The checkpoint stores the option and older checkpoints default to disabled.
The ONNX exporter records the flag; this candidate has not been exported or
benchmarked in a browser. 59 automated tests passed, including gradients,
direction reversal, visibility masks, checkpoint roundtrips, legacy loading,
training and existing export helpers.

## Results on the same development validation examples

| Actual label | Baseline raw top label | Candidate raw top label | Candidate score | Accepted at .8? |
| --- | --- | --- | ---: | --- |
| HELLO | HELLO | HELLO | .668 | No |
| J | J | J | .998 | Yes |
| Z | Z | Z | .997 | Yes |
| Z | Z | Z | .998 | Yes |
| BACKGROUND | HELLO | BACKGROUND | .722 | No sign emitted |

Top-label accuracy increased from 4/5 to 5/5, but positive acceptance remains
3/4. It is misleading to describe the raw 100% as reliable live recognition.
The model was selected by validation macro F1 and loss, not by a validated live
acceptance objective.

## Altered-input probes

The HELLO candidate now classifies frozen first/middle frames as BACKGROUND,
and a reversed HELLO clip as BACKGROUND. These are useful movement-sensitivity
checks, not labeled negative test accuracy. Other gaps remain: a partial HELLO
has HELLO as its top label, and some frozen/shuffled J/Z probes retain their
original label. The full acceptance pipeline still requires real motion, but
these probes do not establish correct trajectory understanding for every class.

## Disposition

Experimental only; do not replace the live checkpoint or lower confidence merely
to accept this one HELLO validation example. There is only one HELLO and one
BACKGROUND validation take, from the same signer, and validation is already
development-known. More reviewed complete-HELLO / held-pose / partial-path /
return-movement examples are needed to evaluate acceptance and select thresholds
without fitting to one example. Negative subcategories were not recorded in the
current metadata and original camera images were not retained. Raw coordinate
validity alone cannot resolve annotation correctness. No additional recording is
required merely to rerun the controlled software checks above.
