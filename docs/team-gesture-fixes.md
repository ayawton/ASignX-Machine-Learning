# Dynamic gesture acceptance: team-requested fixes

Source request: dad.txt supplied by the user on 26 October 2026. Scope: isolated
dynamic letters and words; no conversational translation.

## What is implemented

- The existing motion gate starts capture, buffers pre-roll and ordered motion,
  waits for a quiet interval, then classifies the complete take once. A timeout
  aborts an overlong gesture. A camera stall clears the capture.
- Both main.py and the standalone `python -m ml.predict_sequence` now use this
  completed-take pipeline. The standalone command previously emitted predictions
  from arbitrary sliding windows.
- A new acceptance guard measures wrist/fingertip movement in image x/y,
  normalizes by palm width, median-filters three consecutive observations, and
  requires sustained movement and substantial excursion on one continuous hand
  track. Depth noise, tiny jitter and one-frame spikes do not qualify. Missing
  hand observations break tracks; separate hands are not joined.
- The guard runs before classification. It is motion evidence, NOT proof of the
  correct J/Z/word trajectory. Existing tracking and confidence checks still apply.
- BACKGROUND, IDLE and NO_GESTURE labels are rejection categories (case-insensitive,
  with spaces/hyphens accepted for NO_GESTURE). Prefer BACKGROUND for new data.
  These labels are also rejected when combining separate sequence models.
- Rejection reasons appear in the main camera overlay and console. Background
  detections do not emit a sign. Each completed movement emits at most once.
- New sequence training has validation-based early stopping: default patience 10
  epochs; `--early-stopping-patience 0` disables it. The best validation checkpoint
  is retained. Existing inverse-frequency class weighting, ordered clips and
  person-group split support remain available.

No existing checkpoints, source recordings or labels were rewritten. A frozen
frame can still fool the raw neural classifier; the acceptance pipeline rejects
the absence of motion before trusting that classifier.

Verification: 81 automated tests passed across acceptance, gesture transitions,
body/temple tracking contracts, model grouping, retraining, motion auditing and
existing F1 helpers. A mocked standalone-camera replay verifies one emission per
completed gesture and fails if rolling-window classification is used. These are
software checks, not a new human camera accuracy benchmark.

Follow-up tracking fix: completed-take inference was checking tracking gaps again
after resampling to 30 model steps. That stretched allowed gaps in short takes
and could conceal excessive gaps in long takes. Gap checks now use the original
sampled observations before resampling. A 12-observation regression with a
two-observation dropout now reaches classification; a long take with a real
five-observation dropout is still rejected. Tracking-coverage errors report hand,
shoulder and face-reference visibility. The follow-up focused suite passed 80
tests; no live-camera success is claimed and no checkpoint retraining was needed.

## Initial thresholds and calibration

The defaults are engineering starting points, not calibrated human accuracy:

| Setting | Default | Meaning |
| --- | --- | --- |
| --motion-threshold | .045 | Existing live motion-start threshold |
| --gesture-min-excursion | .35 | Minimum displacement span in palm widths |
| --gesture-min-motion-seconds | .12 | Minimum duration of sustained movement |
| --sequence-confidence (main.py) / --confidence (standalone) | .8 | Class score acceptance threshold |

The guard counts movement above .3 palm widths/second. Its constructor also
allows this speed to be tuned. Tune thresholds using real positive and negative
validation recordings; freeze them before final testing. Do not lower tracking
requirements just to force a predicted answer. Slow/small valid movements may
require different thresholds; live sensitivity has not been measured yet.

The standalone CLI retains --stable-windows/--release-windows as legacy arguments
but reports that they are ignored. Completed-take capture replaces rolling-window
counts. Press R to reset capture and Q/Esc to exit.

## Evidence from the current pilot

The validation-only diagnostic in
`output/team-gesture-review-20261026/motion-audit/report.md` uses four saved clips
(one HELLO, one J, two Z), all from one signer. It bypasses live acceptance checks:

- HELLO's first frame repeated across the clip is accepted as HELLO at ~99.6%.
- J's first frame repeated across the clip is accepted as J at ~89.4%.
- All four shuffled sequences are accepted as their original labels.

These are altered-input probes, not human-labeled negative test accuracy. They
show that high confidence and correct predictions on complete clips do not
establish that this checkpoint learned the required movement order. The saved
test split was not evaluated for this diagnostic.

## Human recordings still required

HELLO-specific correction: the earlier "stop halfway" negative instruction is
withdrawn. Short salutations and greeting waves can be valid HELLO. Do not label
them BACKGROUND just because movement is shorter than a demonstration. Review
their content under docs/vocabulary-v1.md; retain ambiguous cases as UNSURE.
The generic partial/wrong-path recommendations below require gesture-specific
review, not automatic negative labels based on duration or truncation alone.

Keep accepted J/Z/HELLO positives. Collect BACKGROUND takes using the same
180-feature, both-hands, 30-step, 15-FPS recording contract and real Person IDs.
Start with about 10 examples per category as a collection pilot, not a sufficient
sample-size claim. Keep category folders separate so annotation intent survives:

| Category folder | Record | Training label |
| --- | --- | --- |
| held_start | J/Z/HELLO starting poses held still | BACKGROUND |
| held_middle | Poses held partway through a sign | BACKGROUND |
| partial | Deliberately stop before the selected gesture is complete | BACKGROUND |
| wrong_path | Reviewed wrong paths or segment orders | BACKGROUND |
| ordinary_motion | Repositioning and unrelated hand movement | BACKGROUND |

Suggested output roots: `ml/data/vocabulary_v1/negatives/<category>`; each studio
take then goes into its BACKGROUND subfolder. Keep the hand visible during held
pose examples; empty camera frames alone cannot teach confusing poses. The current
recorder requires visible hands/body references, so true no-hand idle periods
should be retained as separate live-rejection tests rather than forced through
the positive recording quality gate.

Do not automatically label every reversed or truncated clip invalid: some altered
movements can be valid ASL variants or other signs. An ASL-knowledgeable reviewer
must define the accepted form and identify genuine hard negatives. Never generate
synthetic identities or claim shuffled/frozen probes are new human recordings.

Before retraining, verify tracking validity and labels for all examples. Use one
person ID consistently across all signs and visits. Keep people separated with
the trainer's --group-plan; filling identity fields alone does not enable that
policy. Preserve an untouched test set. A single-signer pilot cannot establish
generalization to other people or both dominant hands.

## Remaining limits

Gesture-specific direction/segment/endpoint checks are not enabled: there are no
reviewed path annotations or representative negative validation data to calibrate
them. A universal endpoint rule would reject valid words, and movement amount
alone cannot reject every wrong or incomplete path. The next learning change is
training a separate candidate on real positives plus reviewed BACKGROUND examples,
then measuring false activations and missed signs in continuous camera streams.

Do not replace the existing models or raise accuracy claims based on this code
change. Restart the Python camera launcher to load the updated logic. Packaged
EXEs require a rebuild to receive these source changes.
