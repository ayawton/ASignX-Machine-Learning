# Temple context: design and future use

The existing 150-feature format already includes nose/chin and body-relative wrists.
Recognizing HELLO away from the head therefore suggests training shortcuts as well
as a missing direct head-distance signal. Extra features alone cannot establish
ASL correctness or guarantee rejection of incorrect location.

The standard `hands + body` format adds two lateral-face temple-region
proxies and explicit hand/head offsets. It is label-independent: future forehead,
head-side, or face-near signs can learn location without adding per-label code.
It complements handshape and whole-hand motion; static recognition is unchanged.
No hard contact rule is applied to HELLO or other labels.

The references are MediaPipe Face Landmarker vertices 127 and 356, interpreted
as approximate lateral temple regions from the official [canonical topology](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj).
They are not anatomical measurements. The existing [face tracker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
already computes these points, so there is no extra face detection pass.

## Versioned layout

- 0?149: exactly the existing `hands_upper_body_xy_v2` feature prefix.
- 150?155: two temple-reference triples: shoulder-normalized x/y and visibility.
- 156?179: eight triples: head-width-normalized x/y offsets and visibility,
  ordered by Left/Right hand, temple 127/356, wrist/index fingertip.
- Contract: `hands_upper_body_temples_xy_v3`, width 180.

The format includes both sides for either signing hand. It never mixes hand,
pose, and face z values; hand-local z remains in the handshape blocks. It adds
neither hips nor a full face mesh to the saved dataset. Missing head references
are masked, and temporal resampling does not invent their visibility.

## Limits and collection

Head turns, occlusion, and small faces can make lateral references unreliable.
New recordings require visible shoulders and temple references in at least the
configured detection fraction (50% by default); live inference rejects long gaps.
Temple references are part of the normal body mode. The older 150-feature
format remains available as `hands + body (legacy 150)` for existing datasets.
Face locations refresh at the existing 20 FPS cap, with brief bounded caching.

Collect valid natural variation plus incorrect-location movement under BACKGROUND:
for example the same HELLO-like trajectory near the chest. The dynamic predictor
suppresses BACKGROUND rather than emitting it. Train and evaluate all labels with
the same feature format, retaining independent held-out recording sessions. Check
both acceptance of valid HELLO and false acceptance away from the head; do not
claim improvement from added landmarks without those recordings.

Existing 150-feature recordings/models remain supported. They cannot be expanded
faithfully into 180 features without original videos. Do not append zeros or mix
formats. Use a new dataset folder for a feature-format change. You may reuse a
checkpoint filename: successful sequence retraining automatically backs up the
previous checkpoint and reports before replacing them. The web SDK does not yet implement either body-aware sequence format.
