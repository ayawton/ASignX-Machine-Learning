# Review the three disputed HELLO outputs

Current HELLO annotation rule: a shortened greeting can still be valid HELLO.
"Stopped halfway" is not an automatic BACKGROUND category. The earlier
instruction to collect partial HELLO negatives was ambiguous. Keep uncertain
cases UNSURE and review affected existing examples without changing originals.

Open ASignX_ReviewHello.bat. It loads the three accepted REST captures from
output/live-diagnostics/20261027T004006Z-14300d, without opening the camera.

For each replay, choose:

- Valid HELLO: an accepted forehead-salute or ordinary greeting-wave form.
- Unrelated movement: an incidental motion that is not a valid HELLO greeting.
- Unsure: the replay is unclear, incomplete, or insufficient for judgment.

Decisions save immediately in that diagnostic folder's label_review.json. The
reviewer advances to the next example. Close the window once all three are
reviewed. Previous/Next and the timeline slider allow revisiting a decision.
Playback runs at half speed; Play/pause or Space toggles playback. H/B/U are
keyboard shortcuts for the three decisions, but the buttons work as well.

The display is reconstructed tracked geometry, not video. Hand scale is recovered
from the saved wrist/index-tip and head-reference offsets when available. Missing
or degenerate geometry is not filled with an estimated hand. The display cannot
prove physical contact or show facial expressions or original visual details.
Choose Unsure if these limitations prevent a reliable annotation.

The 0/REST key in the diagnostic was an intention marker, not a recognition mode.
Being captured in REST does not automatically make an actual greeting wave a
negative example. Do not use head contact alone as the decision rule.

No source file, training label or checkpoint is changed by review. Later label
changes require a reviewed dataset snapshot and new evaluation provenance; they
must not silently overwrite the existing evaluation collection.

## Review saved evaluation BACKGROUND labels

ASignX_ReviewBackground.bat opens all 15 BACKGROUND files in the evaluation
collection. It hides model predictions and filenames in the replay header.
Choose Valid HELLO for accepted salutations or greeting waves; choose Unrelated
movement only for a clear non-gesture/negative example. A held pose is not a
complete dynamic greeting. If it could be another supported sign, or geometry
is unclear, choose Unsure rather than forcing it into BACKGROUND.

This mode uses saved prepared sequence steps with uniform playback across the
stored take duration; original per-frame timestamps were not retained. It is
for reviewing movement content, not diagnosing exact timing or camera speed.

Decisions are saved separately in
output/label-reviews/evaluation-background-20261027/label_review.json. Source
files, existing labels, training splits and checkpoints are unchanged. This
review audits annotation consistency after accepting wave variants; it is not
a new accuracy benchmark or automatic permission to retrain on evaluation data.
