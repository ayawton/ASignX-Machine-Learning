# ASignX vocabulary v1 and collection checklist

Scope: recognize one isolated ASL letter or word sign at a time. Sentence
translation is outside this version. This is a collection and release plan;
it does not change an existing checkpoint's supported labels.

## Target vocabulary: 34 sign classes

| Group | Exact labels | Count |
| --- | --- | --- |
| Static letters | A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y | 24 |
| Dynamic letters | J, Z | 2 |
| Held word sign | LOVE | 1 |
| Moving word signs | HELLO, THANK_YOU, PLEASE, SORRY, YES, NO, HELP | 7 |

Use these exact uppercase labels for new recordings. THANK_YOU is one sign
class despite its two-word English display name. LOVE means the crossed-hands
over-chest sign, not the I-love-you handshape. Held means the target pose can be
evaluated during a hold; entering and leaving that pose still involves movement.

Reference demonstrations:
- ASL University vocabulary: https://www.lifeprint.com/asl101/pages-layout/signs-listedbylesson.htm
- LOVE: https://www.lifeprint.com/asl101/pages-signs/l/love.htm
- THANK_YOU: https://www.lifeprint.com/asl101/pages-signs/t/thankyou.htm
- HELP and its variations: https://www.lifeprint.com/asl101/pages-signs/h/help.htm

Agree on one reference form per word with a knowledgeable ASL signer before bulk
recording. Document accepted variations. Camera visibility checks cannot determine
whether the sign itself is correct.

### Current pilot form policy (27 October 2026, latest user instruction)

The user narrowed the current test to forehead-salute HELLO. With
`--gesture-form-checks`, J requires sustained extended-pinky evidence, Z requires
sustained extended-index evidence, and HELLO requires an open hand beginning
near a forehead/temple proxy. The temporal model still selects the movement;
these necessary checks do not establish correct complete J/Z trajectories.
Neither global screen-right motion nor literal forehead contact is inferred.
Landmarks can establish proximity, not physical touch. Test both valid signs
and wrong-handshape movements before considering this policy validated.

The checks are enabled in ASignX_TestAllSigns.bat and
ASignX_TestReviewedHello.bat. Checkpoint weights and earlier annotations are
preserved. Do not relabel prior greeting waves as BACKGROUND automatically.
This narrower pilot policy is not a claim that other ASL HELLO forms are invalid.

### Earlier wave-inclusive HELLO policy (superseded for this pilot test)

HELLO includes both the recorded forehead-salute form and an ordinary greeting
wave. Head contact is not an acceptance requirement for this class. Do not add a
salute-only head-location gate or label a valid greeting wave BACKGROUND merely
because it lacks head contact. Review held poses, partial/reset motions and
incidental movements against this broader definition; a shortened salute that
is itself a valid greeting wave is not automatically a negative example.

Both forms use the same HELLO label. Variant annotations can be retained for
evaluation, but existing files must not be automatically relabeled. This scope
decision does not prove that the current checkpoint recognizes every valid wave.

Annotation clarification: do not collect or label "HELLO stopped halfway" as
BACKGROUND based on shortening alone. A short salute or wave can still be a
valid HELLO. That earlier collection instruction was ambiguous and is withdrawn
for this class. Use clearly stationary poses or unrelated non-greeting motions
as negatives; keep ambiguous motions UNSURE. Previous datasets containing a
"partial HELLO" batch need content review before labels or scores are trusted.
Do not automatically relabel all such files: their subtype was not recorded.

## Collection stages

1. Pilot: J, Z, HELLO. Record 10 accepted takes of each from one signer to check
   recording, labels, and storage. This is a workflow check, not an accuracy target
   or an independent benchmark. Do not train or replace a deployed model yet.
2. Alphabet baseline: audit the existing 24-letter static data and collect missing
   J/Z coverage. Test all 26 letters together with the selected checkpoints.
3. First words: HELLO, THANK_YOU, LOVE, HELP. These exercise body location and
   one-/two-hand behavior. LOVE needs a combined hand/body recognition path;
   the present per-hand static letter classifier is insufficient by itself.
4. Expansion: PLEASE, SORRY, YES, NO, followed by a new vocabulary version only
   after checking recognition on the previous vocabulary too.

## First recording session

- [ ] Open ASignX_DynamicStudio.bat. The current source supports manual Start/Save.
- [ ] Choose a fresh output folder: ml/data/vocabulary_v1/pilot/sequences.
- [ ] Set Hand mode to both and Landmarks to hands + body (180 features).
- [ ] Keep Sequence steps at 30 and Sampling FPS at 15 across this collection.
- [ ] Use a 3-second preparation countdown. Keep the maximum duration at 10 seconds
      as a safety limit; save immediately after the complete sign rather than
      waiting out the limit. Each take must contain only one performance.
- [ ] In Recording identity, set Person ID to P001 for the first real signer.
      Reuse that same person ID for all their signs and later visits.
- [ ] Set Recording visit ID to P001_visit01 and enter an actual Device ID.
- [ ] Leave Reserved split blank for this pilot. Keep pilot data separate from
      the final evaluation collection.
- [ ] Set label J. Record, check the preview, save, and repeat until 10 takes are
      accepted. Discard/re-record incorrect or clipped performances.
- [ ] Close the camera to change label. Repeat for Z and HELLO.
- [ ] Confirm 10 saved .npz files under each label folder: 30 total.

The studio saves landmark arrays and metadata, not raw camera videos. A saved
file proves storage and minimum tracking quality, not signing correctness.
Retain source demonstration videos separately when human review or later
re-extraction is required. Do not mix old 63/126/150-feature recordings with
the new 180-feature collection. Do not rename existing Hello examples in place;
review legacy labels and provenance before importing them.

## After the pilot

- Inspect saved files for labels, shape (30, 180), tracking gaps, and person IDs.
- Have the signing form reviewed before scaling collection.
- Recruit different signers and assign each person to training, validation, or
  test before bulk collection. Keep all of a person's signs and visits in that
  one partition. A possible initial plan is 6/2/2 people; it is not a guarantee
  of sufficient coverage. If fewer people are available, report that limitation.
- Use the trainer's explicit --group-plan for person-separated evaluation.
  Filling identity fields alone does not activate that training split policy.
- Collect rest, repositioning, incomplete signs, and unrelated movements in a
  separate negative-example collection. Do not label these as valid J/Z/words.
  Decide which are training BACKGROUND examples and which are reserved rejection
  tests before training; the current selected model has no background class.
- Save a new candidate checkpoint and compare old and new classes before changing
  the recognition launcher. Keep the previous checkpoint available.

## Terms

Class: an answer category such as J. Label: the answer attached to a recording.
Take: one complete recording. Dataset: all examples and their labels.
Checkpoint: saved learned model (.pth). Negative example: input that should not
trigger a supported sign. Training: learning from examples. Validation: choosing
and tuning a model. Test: a reserved final check, ideally on different people.
