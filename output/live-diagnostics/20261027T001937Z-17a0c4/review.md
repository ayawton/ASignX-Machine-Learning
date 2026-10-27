# Completed focused diagnosis

The user identified this latest run as completed. The immediately preceding
20261027T001823Z-d6d801 run was incomplete and is excluded from interpretation.
The checkpoint and decision thresholds match the earlier training-v2 run; the
static alphabet model is no longer loaded.

## Observations

- HELLO section: no HELLO emissions, seven Z emissions. Two additional captures
  have HELLO as their raw top label below .8; three have low-confidence Z.
- REST section: no sign emissions. There are normal BACKGROUND, insufficient
  motion, short-capture and below-confidence outcomes. Hand visibility is 49.1%
  over the block, so this exercises more visible-hand activity than the earlier
  no-hand REST test, but does not prove every negative category was captured.
- HELLO throughput is 13.83 FPS (overall 14.50) versus 18.89 FPS in the earlier
  longer HELLO section. Median detector time is 63.76 ms in HELLO. Face references
  and shoulders remain visible throughout; hands are visible on all processed
  frames of the captured HELLO spans. No capture interval exceeds 200 ms.
- Most captured HELLO spans last about .37-.58 seconds after the motion-start
  trigger, before adding the pre-roll observations. Earlier successful/uncertain
  spans varied from about .45 to over 1.3 seconds. This suggests checking the
  actual captured model inputs, but does not prove an incorrect boundary.

This run contradicts a claim of reliable HELLO recognition. Timing and target
labels alone cannot distinguish changed signing, segmentation, sampling effects
or classifier sensitivity. No camera images or exact input arrays were captured
in this run, so faithful offline reproduction is not possible from these logs.

## Next diagnostic capability

main.py now has an opt-in --capture-gesture-inputs flag requiring a diagnostics
directory. The current training-v2 diagnostic launcher enables it and states
that completed landmark inputs are saved locally. Default metric-only logging
remains unchanged when the flag is absent. Camera video is not retained.

Each completed take is saved under gesture_inputs/ with original timestamps,
input arrays, explicit missing-observation flags, target block, captured scores
or rejection, and checkpoint hash. ml.replay_gesture_inputs can replay the
preparation/classifier stages against the same checkpoint without a camera.
This enables investigation of a failure without repeatedly collecting new
training batches. Captured diagnostic takes are not automatically training data.

No threshold or checkpoint was changed for this addition. 31 automated tests
passed across tracing, replay, diagnostics, transitions, grouping and display.
