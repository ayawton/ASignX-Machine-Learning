# Reviewed training candidate: not promoted

The user reviewed all 24 potentially ambiguous BACKGROUND training/development
examples: 23 remain BACKGROUND and one is UNSURE. The unsure example was in the
training partition and was excluded from this separate copied dataset. No review
changed a training label to HELLO.

The resulting collection has 76 takes: 30 HELLO, 27 BACKGROUND, 9 J and 10 Z.
Training has 54 takes; validation and reserved test retain their existing 11 takes
each. The reserved test and separate evaluation files were not added to training.
All original source files, review decisions and the current checkpoint were
hash-checked and preserved. No test inference was performed.

The candidate uses the same architecture, seed, training settings and .8 class
cutoff as the current model. Best epoch 10 was retained and training stopped at
epoch 20. The only dataset change was exclusion of the uncertain example.

## Comparison using the recorded evaluation review

| Model | HELLO accepted correctly / 9 | BACKGROUND rejected / 11 | Remaining HELLO error |
| --- | ---: | ---: | --- |
| Current training-v2 | 8 | 10 | Abstains on 1 |
| Reviewed-training candidate | 8 | 10 | Incorrectly accepts Z on 1 |

This is not a useful improvement. Keep the current checkpoint selected; no new
launcher or default-model change is warranted. The cleaned candidate remains an
experimental artifact with its full preparation and results saved.

The evaluation judgments were recorded before the latest explicit discussion of
short HELLO variants; do not assume perfect annotation correctness. These are
same-person, already inspected development files, not a final independent test.
The landmark review cannot recover original appearance or independently verify
ASL correctness. Further tiny tuning runs on these examples would not establish
generalization. The next validation should use a clearly identified fresh signer
or recording session with explicit accepted HELLO variants and true non-greeting
negatives, while keeping the checkpoint/settings fixed.
