# Reproduced sampling fault and bounded timestamp correction

This run captured 50 completed segments. Exact replay against the captured
checkpoint reproduces all 39 segments that have a recorded raw prediction;
the other 11 contain no logged raw prediction because a guard/background check
rejected them. Original replay results are in replay.json. The checkpoint hash
matches the capture. Neither the model nor the .8 confidence cutoff changed.

The HELLO test block processes about 13.77 frames/second while the checkpoint
sampling rate is 15. The old completed-take path inserts zero feature rows for
missed nominal sampling ticks. For example, take_000087 has 12 actual observations
over .8 seconds but becomes 13 sampled rows with 3 synthetic all-zero rows.
Those zeros mean missing hand/body context to the classifier even though actual
processed frames contain visible hands. This is different from observed tracking
loss. sampling-comparison.json records the exploratory observed-frame comparison.

## Fix

Completed body-context takes now interpolate coordinates onto the checkpoint's
sampling clock using original capture timestamps. Interpolation requires visible
endpoints for each hand/reference; real missing detections remain masked. Camera
intervals over the existing permitted .2 seconds reject the take instead of
inventing long motion. Tracking coverage, internal gap checks, minimum take
length, sustained-motion guard, .8 confidence cutoff and model weights remain.
The raw hand-only streaming path is unchanged.

## Same-input replay outputs

| Preparation | HELLO outputs | Z outputs | No sign output |
| --- | ---: | ---: | ---: |
| Original completed-take sampling | 1 | 18 | 31 |
| Corrected timestamp interpolation | 22 | 0 | 28 |

These are segment outputs, not human-labeled accuracy. Segments can include
repositioning or incomplete gestures even though the overall block is HELLO.
The corrected raw top labels are 28 HELLO, 20 BACKGROUND and 2 tracking/length
rejections. Six top-label HELLO predictions remain below .8. Do not lower the
cutoff solely to accept them or claim that every accepted HELLO is complete.

77 automated tests passed, covering slow-camera interpolation, exact timestamps
on large clocks, true missing detections, camera stalls, input immutability,
inference, diagnostics, replay, motion acceptance and existing transitions.
Fresh live behavior after this fix is unverified. Existing training/evaluation
recordings and checkpoints remain separate and unchanged.
