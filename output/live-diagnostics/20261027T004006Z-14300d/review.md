# HELLO acceptance and visible-hand negatives after sampling fix

## User review outcome

The user reviewed the three disputed REST outputs with the landmark replay tool:
take_000470 and take_000622 are marked HELLO; take_000672 is marked UNSURE.
All reviewed source hashes still match. The first two outputs agree with the
user's accepted greeting-wave/salute definition. The third remains unresolved
and must not be counted as either correct or a confirmed false positive, or
used as a supervised training label. There are no confirmed BACKGROUND examples
among these three reviewed outputs. This selected three-example review does
not establish overall recognition accuracy or unknown-movement rejection.

See label_review.json and human_review_summary.json. Original recordings and
checkpoints are unchanged; none of these clips was automatically added to
training or used to tune a threshold.

## Recorded diagnosis

This run uses timestamp_interpolation_v1 with the same training-v2 checkpoint,
.025 motion-start threshold and .8 class-confidence cutoff.

- HELLO section: 5 accepted HELLO outputs, 1 HELLO candidate below .8, and
  3 BACKGROUND decisions. These are captured segment counts, not attempt accuracy.
- REST section: 3 accepted HELLO outputs (.931, .883, .884) and one Z candidate
  at .468. That Z was NOT accepted. The three HELLO outputs occurred in the
  user-marked negative block, but the user subsequently confirmed that ordinary
  greeting waves should also be accepted as HELLO. They are not established
  false activations without reviewing whether the movements were greeting waves.
- Hand visibility in REST is 97.1%, making this a meaningful visible-hand test.

The display previously used MOTION for both accepted predictions and raw guesses
below threshold. Low-confidence candidates now display Uncertain instead; this
display change does not change the three high-confidence HELLO decisions.

## Vocabulary definition needed

The user describes starting HELLO by touching the head. ASL University also
recognizes natural greeting-wave forms of HELLO/HI:
https://www.lifeprint.com/asl101/pages-signs/h/hello.htm

Consequently, absence of head contact alone is not sufficient to label every
wave invalid ASL. The user confirmed that this version should accept both the
recorded forehead-salute form and ordinary greeting waves. No head-contact gate
should be added. The 0 key annotates test intent; recognition continues normally.
Retain current files and review annotation consistency under the accepted variants
instead of automatically relabeling clips based on model outputs or block keys.

## Geometry diagnostic (not contact detection)

The recorded features include two-dimensional hand/head offsets normalized by
head width, not physical contact measurements. The three accepted REST takes
have first-visible index-tip distances to the nearer lateral-face reference of
about 1.30, 1.17 and 1.58 head widths. Their minimum distance within the first
40% of the captured observations is about .02, .20 and .67 respectively. Thus
some approach the head region later in the capture. This supports checking
ordered starting-region and movement conditions for a defined variant; it does
not establish actual head contact or ASL correctness.

The corresponding accepted HELLO starts are near the head region, but a rule
must account for capture pre-roll, movement into position and valid variants.
No new acceptance rule, confidence change or retraining was applied in this
review. Keep the original inputs for replay and reviewed subtype annotation.
