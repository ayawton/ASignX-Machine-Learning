# Reviewed live HELLO candidate

Experimental only; the existing training-v2 and combined recognition launchers
still select their earlier checkpoints. Use ASignX_TestReviewedHello.bat for a
fresh same-person trial of this candidate. No accuracy claim follows from fit.

## Diagnosis

The fresh-session human review identified 13 HELLO salutes and 5 ordinary return
movements. The earlier model accepted 3 of the 13 reviewed salutes and rejected
all 5 return movements. All HELLO-block frames had tracked hand/shoulder/face
references; the run had no capture intervals exceeding 200 ms. Missed captures
had a median wrist movement extent of 0.328 shoulder widths, compared with 0.349
for accepted captures. Their captured duration medians were 0.783 and 0.831 s.
The training HELLO clips had a median stored duration of 1.728 s. These durations
include holds and preparation; they do not measure signing speed alone. Hand
positions also varied across the sets. Tracked geometry cannot independently
establish full ASL correctness or whether every desired start frame was captured.
Human review supplies the positive/negative decisions here.

See the source run's capture_comparison.json and capture_comparison.png for
geometry measurements. No capture thresholds or model architecture were changed.

## Data and training

Started with the previous reviewed dataset snapshot, which excluded one unsure
BACKGROUND training clip. Imported all 18 reviewed HELLO-block captures, with
source and review hashes, using the exact live timestamp sampling and body-context
resampling contract. Prepared-input prediction equivalence was checked for every
import. All imports were assigned to training, with the original visit and signer
identity retained. Existing validation/test membership was preserved. Original
test was not evaluated. The new snapshot has 94 samples: 43 HELLO, 32 BACKGROUND,
9 J, and 10 Z; 72 train, 11 validation, 11 reserved test.

Training used the existing configuration and seed, with strict fixed splits.
Best epoch 10; early stopping completed 20 epochs. Source hashes and the active
checkpoint were verified unchanged after training and replay.

## Comparison

| Development check | Earlier model | Candidate |
| --- | --- | --- |
| Reviewed salutes imported into training | 3/13 accepted | 13/13 accepted |
| Reviewed return movements imported into training | 5/5 rejected | 5/5 rejected |
| Older reviewed HELLO development clips | 8/9 accepted | 8/9 accepted |
| Older reviewed BACKGROUND development clips | 10/11 rejected | 10/11 rejected |
| Previously accepted J/Z outputs in fresh-session replay | 10/10 retained | 10/10 retained |

The first two rows are training-fit checks, not held-out recognition accuracy.
The old external evaluation clips have been repeatedly used as development data.
The previously missed development HELLO now has HELLO as its top class but stays
below the unchanged 0.8 threshold (about 0.689). The same older background clip
still falsely emits HELLO (about 0.940). Raw scores are not calibrated accuracy.
J/Z replay retains five J and five Z accepted outputs, with no extra outputs in
those captured segments. Replay checks classification, not automatic boundaries.

A fresh live session is required before considering replacement of the active
model. Use normal-speed salutes, J/Z checks, and visible-hand rest/repositioning.
No wave recognition or performance on other people has been established.
