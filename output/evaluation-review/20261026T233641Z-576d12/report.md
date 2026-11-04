# New HELLO / BACKGROUND evaluation

20 newly supplied takes; fixed models and confidence threshold .8. No retraining.

| Model | HELLO accepted correctly / 5 | Other sign outputs on HELLO | False sign outputs on BACKGROUND / 15 | Tracking rejections HELLO / BACKGROUND |
| --- | ---: | --- | ---: | --- |
| positive_only | 5 | {} | 15 | 0 / 0 |
| background_baseline | 0 | {} | 3 | 0 / 0 |
| motion_features | 0 | {} | 5 | 0 / 0 |

## Interpretation limits

These counts evaluate prepared files, not live automatic capture. Tracking rejection can make a negative appear safely rejected without demonstrating learned rejection. Consult raw top labels and decision reasons in report.json. Categories of negatives are not separately annotated. The recordings are from the same reported signer and have no exact feature overlap with prior pilot datasets. Original recordings and checkpoints are unchanged.
