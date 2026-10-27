# Pilot motion-start comparison

Baseline: ../20261026T142831Z-0d89cc/summary.json, motion start .045.
Second session: summary.json, motion start .025. Both use the same checkpoint,
confidence threshold .8, excursion minimum .35 and movement duration minimum .12.

| Intended block | Baseline accepted outputs | Second-session accepted outputs |
| --- | --- | --- |
| J | J: 5 | HELLO: 1; J: 0 |
| Z | Z: 6 | Z: 7 |
| HELLO | HELLO: 1; Z: 1 | HELLO: 4; Z: 4 |
| REST | none | Z: 1 |

These are event counts, not recognition accuracy: individual attempts were not
annotated and the sessions contain different performances. Do not infer that
changing the threshold alone caused all differences.

The second J block reached classification eight times: seven below-threshold
results and one accepted HELLO. Its problem is not simply failing to start capture.
The HELLO block reached classification nine times: one low-confidence Z, four
accepted HELLO results, and four accepted Z results. The accepted events alternate
HELLO then Z. This is consistent with separate classification of return/reset
movement, but raw trajectories/video were not logged, so that explanation remains
unverified. The REST Z score was .975: a confidence cutoff of .8 does not solve
unknown-movement rejection for this checkpoint.

Observed overall throughput was 15.42 FPS versus 15.48 in the baseline; neither
session logged a capture interval over 200 ms. Shoulders were visible in 99.7%
and face references in 100% of second-session frames. HELLO whole-block hand
visibility was 92.8%. These measurements do not establish absence of blur or
correct ASL signing, but provide no evidence of a major processing slowdown.

## Decision and next data

Do not promote the .025 setting as a general fix. The regular diagnostic launcher
remains at .045; the sensitive launcher remains a separate development comparison.
Keep the positive recordings and both logs. Do not lower confidence or retrain
again on only the same positive clips.

Next collect a small real BACKGROUND batch with the same both-hand, 180-feature,
30-step, 15-FPS contract. For an initial ten-take collection check:

- 3 takes: hold the starting handshapes for J, Z and HELLO without performing them.
- 4 takes: ordinary return-to-rest/return-to-start movements, excluding the valid
  sign movement itself from each recording.
- 3 takes: unrelated repositioning while hands and body references remain visible.

Use label BACKGROUND and Person ID pilot_user_001, and a fresh output folder such
as ml/data/vocabulary_v1/negative_pilot. Keep the same real signer ID across all
their positives and negatives. Stored landmarks cannot establish correctness;
review that none of these takes is actually a valid supported sign. Keep empty
camera periods as separate rejection tests because the recorder requires visible
hands. Ten negatives are a workflow starting point, not sufficient evidence of
reliable general rejection. Expand to reviewed partial/wrong paths and different
people before release evaluation. Preserve held-out examples for both positives
and negatives rather than optimizing only positive acceptance.
