# Four-class pilot review: do not promote

The user supplied 10 BACKGROUND recordings. Nine pass the same tracking checks
as the 27 previously accepted positives; one background take has excessive
tracking loss. All retained recordings are 30 x 180 features at 15 sampling FPS.
Copied examples have one user-reported signer ID; source metadata had no signer
IDs. Source recordings and existing checkpoints were preserved and hash-checked.

The candidate was trained from scratch with the existing TCN-attention
architecture and early stopping. Previous positive train/validation/test
assignments were preserved. Background takes use seeded 7/1/1 assignments.
Training stopped at epoch 25 and retained epoch 15.

## Development validation (five takes)

| Actual label | Raw prediction | Score | Result at threshold .8 |
| --- | --- | ---: | --- |
| HELLO | HELLO | .670 | Rejected: misses known sign |
| J | J | .965 | Accepted |
| Z | Z | .987 | Accepted |
| Z | Z | .984 | Accepted |
| BACKGROUND | HELLO | .700 | Rejected by confidence; wrong raw class |

Raw classification accuracy is 4/5, but that hides the live missed HELLO and the
failure to correctly identify the held-out BACKGROUND. At the live confidence
threshold the candidate accepts only 3/4 positive validation takes. The raw
HELLO score on the background exceeds the score on the real HELLO, so a global
confidence reduction cannot correctly accept the latter while rejecting the
former in this comparison.

The development audit also finds all six training HELLO examples below .8
(.611-.750). Six of seven training backgrounds have BACKGROUND as their raw top
class; one has HELLO at .498. This is not simply failure to load a background
class: the loaded checkpoint has all four labels, with explicit background
suppression. It has not learned adequate HELLO/background separation.

## Disposition

Do not replace the prior model or present this as a successful rejection model.
The candidate and its launchers are retained as experimental artifacts only.
No live test was run and the saved test split remains unevaluated. This is a
single-signer pilot; the one negative validation example is insufficient to
estimate general rejection performance.

Next investigate movement representation and the actual HELLO/background
recording content. The earlier frozen-frame audit already demonstrated pose
shortcuts in the three-class model. Do not repeatedly tune the confidence cutoff
or collect another arbitrary batch without determining which movement/pose
distinctions are missing. File validity alone does not verify signing correctness
or distinguish the negative subcategories, which were not saved in metadata.
