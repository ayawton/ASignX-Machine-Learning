# Expanded training-v2 pilot

The new collection contains 22 HELLO and 20 BACKGROUND takes. All HELLO and
19 BACKGROUND takes pass the prepared tracking checks. The rejected source take
is retained unchanged. The 41 usable examples were added to the previous 36
copies: 30 HELLO, 28 BACKGROUND, 9 J and 10 Z (77 total).

Existing take partitions were preserved and new training examples were split
with seed 42: total 55 train / 11 validation / 11 reserved test. All recordings
belong to the same user-reported left-handed signer. No signer-independent score
is claimed. The saved test set was not evaluated.

The architecture and training options match the previous motion-feature pilot:
TCN-attention with signed movement features, up to 50 epochs, early-stopping
patience 10. Best epoch 12 was retained; training stopped after epoch 22.

## Same separate development evaluation collection

The 5 HELLO and 15 BACKGROUND evaluation takes were not added to training. Their
file hashes and the original checkpoints are preserved. The confidence cutoff
remained .8. No threshold search or additional training run was performed.

| Model | HELLO accepted correctly / 5 | Negatives without a sign output / 15 | Incorrect sign outputs on negatives |
| --- | ---: | ---: | --- |
| Previous motion-feature pilot | 0 | 10 | 5 Z |
| Expanded training-v2 | 5 | 11 | 4 HELLO |

No tracking rejections account for these outcomes. All 11 internal validation
takes have the correct raw top class; all 7 positive validation takes pass the
confidence cutoff and all 4 validation backgrounds emit no sign. These internal
results do not override the four mistakes on the separate evaluation batch.

## Disposition and live check

The candidate is suitable for an experimental live comparison, not promotion as
a reliable completion detector. Four negative files are confidently accepted as
HELLO. Their negative subcategories are not separately annotated; do not infer
the categories from predictions or file order. No recognition confidence cutoff
was lowered to produce the positive result.

Run the project-root ASignX_DiagnoseTrainingV2.bat for J/Z/HELLO and REST sections
using keys 7/8/9/0, then Q. It keeps the last sensitivity comparison's .025 start
threshold, .35 minimum excursion, .12-second minimum motion, and .8 class score
threshold. Only the dynamic checkpoint changes relative to that comparison.
The default recognition launcher and earlier models are unchanged.

These are same-signer, development-known prepared-file results. Actual camera
capture, segmentation and false activations still require live measurement.
Original footage is not stored, so coordinate validity does not verify ASL
correctness or negative annotations. Once results are used for development,
collect fresh data before making final test or new-person claims.
