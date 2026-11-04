# Evaluation disposition

All 20 new evaluation files (5 HELLO, 15 BACKGROUND) have the expected 30 x 180
feature contract, 15 sampling FPS, and unique take IDs/features. There is no exact
feature overlap with the previous three pilot datasets. All pass the current
prepared-clip tracking checks. Sources and model weights were hash-checked and
preserved. No training, threshold tuning or saved test evaluation was performed.

## Results

The positive-only checkpoint accepts all 5 HELLO clips but also predicts signs on
all 15 negative clips. The background baseline accepts no HELLO clips and emits
sign labels on 3 negatives. The motion-feature candidate also accepts no HELLO
clips and emits Z on 5 negatives. Thus neither background-aware candidate is
ready to replace the live model. The apparent raw 5/5 result on the earlier tiny
validation set did not establish reliable separation on this new collection.

These are prepared-file classifier decisions, not live false-activation rates:
the live motion guard could reject stationary negative examples. No tracking
rejections account for the counts in this batch, so learned class separation,
confidence and recording-content differences need investigation.

## Negative categories

All five false Z outputs from the movement-feature candidate occur on the five
most recently saved negative files. The previous ten have no accepted sign
outputs, although one has HELLO as its low-confidence raw top label. File order
alone does not establish whether those last five were incomplete signs or reset
movements. The user has been asked which category was recorded last. Do not
invent subtype annotations or relabel clips using model predictions.

Keep this evaluation collection separate from training. If it is later used for
development or training, record that change and collect fresh evaluation data
before making new generalization claims. Original video is unavailable here,
so numeric validity is not a review of ASL correctness or completeness.
