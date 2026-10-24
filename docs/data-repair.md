# Source and preprocessing investigation

This investigation preserves the original CSV, checkpoints and historical splits. It fixes source traceability and validation safeguards; it does not claim an F1 improvement or silently change uncertain labels.

## Recovered evidence

`output/data-repair/quality-report.json` identifies 24 hashed import sessions (96,145 rows) with exact session-hash matches to `C:\Users\jibbi\Downloads\archive (4).zip`. The present-day archive content hash is recorded. This establishes the source tag used by the importer, not a signer identity or a unique image for every row. Legacy source strings and other unknown identities remain unresolved.

The source archive contributes these conditions to the frozen benchmark:

- Training: A, B, C, D, E, F, H, L, N, P, R, S, T, U, V, Y; 64,409 rows.
- Validation: W; 4,165 rows.
- Test: G, I, K, M, O, Q, X; 27,571 rows.

The test labels listed above have no training examples of the same label from that archive. This is a concrete source/class coverage gap. It does not prove the test labels are wrong. Moving those rows into training would invalidate the historical comparison, so the benchmark is preserved.

For row 110182, the search found a geometrically matching image at:

`ASL_Alphabet_Dataset/asl_alphabet_train/M/M (100).jpg`

Re-extraction agrees with all stored coordinates within approximately 7.45e-9 and also agrees on Right handedness. See `row-110182-recovery.json`, `row-110182-original.jpg` and `row-110182-landmarks.jpg` under `output/data-repair/`. This rules out a numerical extraction mismatch for this example; it does not verify the linguistic label or landmark anatomical correctness. No M-to-N relabeling was performed.

## Safeguards implemented

- New extraction audits bind accepted output CSV line numbers to original image hashes and paths, including ZIP members. They also record the committed dataset hash and a hash of the stored feature values.
- Stable source-session hashing is shared between import and recovery code.
- Split validation rejects identical canonical features carrying conflicting labels, including contradictions inside the training split.
- New static training manifests record and validate the actual handedness preprocessing contract. Older manifests remain readable for historical comparisons.
- Source audits record coverage gaps and unresolved identities without inventing people or altering splits.

## Reproduce the investigation

```powershell
.\.venv-py314\Scripts\python.exe -m ml.recover_image --row 110182 --archive "C:\Users\jibbi\Downloads\archive (4).zip" --max-images 1000
.\.venv-py314\Scripts\python.exe -m ml.data_quality --source "C:\Users\jibbi\Downloads\archive (4).zip"
.\.venv-py314\Scripts\python.exe -m pytest -q
```

Open `output/data-repair/review.html` for the source coverage table and the recovered original image alongside its landmarks. `session-provenance.csv` is bound to the current dataset hash. `benchmark-plan-template.csv` groups the recovered archive globally across labels and leaves unknown identities and split choices blank. It is deliberately incomplete and cannot yet be passed to training.

## Remaining data work

Label and handedness corrections require original evidence and competent sign review. Archive identity cannot establish signer independence. The next benchmark needs verified origins/signers across all 24 labels, with entire groups reserved before training. Original difficult examples remain valuable evaluation evidence. Do not remove them just because a model misclassifies them.

The most recent candidate's historical test macro F1 remains 56.49%, versus 59.49% for the incumbent. Validation success alone does not justify replacing it. No additional retraining or replacement is performed by these repair tools.
