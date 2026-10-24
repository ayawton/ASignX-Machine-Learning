# Exhaustive legacy source-image replay

Completed every eligible color image in archive (3) and every static-letter image in archive (5), using the current importer retry path. All 63 stored float32 coordinates must reproduce within 1e-6, with the same letter and handedness. No tolerance was widened to increase coverage.

| Archive | Images examined | Current rows reproduced |
|---|---:|---:|
| archive3 | 65,774 | 15,735 |
| archive5 | 24,000 | 11,281 |

Added row-level source evidence for **27,016** previously unresolved rows. Now **122,708 / 159,534** current rows have import, capture or image-replay source support. Added 1,603 conservative session co-location constraints; 1,648 sessions now have a source-family constraint.

A replay match establishes compatible stored geometry in a source container. It does not uniquely identify a photo, prove that a label is linguistically correct, or identify a person. All matching images and subject-folder aliases are retained. Unmatched rows remain unresolved; an exhaustive pass through these archives with this extractor is not proof that their originals do not exist. Different historical detector settings, tracking state or preprocessing can prevent exact replay. A partially matched artificial session receives a conservative source co-location constraint, not an assertion that every row has a verified origin. Signer IDs and verification status were not fabricated.

The original 159,534-row training CSV and existing model weights were unchanged. See the separate [source-separated training experiment](../../source-separated-training/20261024-source-holdout/report.md) for the evidence-only subset, all-row exclusion ledger, fixed split and new candidate.

[Every image/row match](image-row-matches.csv), [recovery summary and hashes](summary.json), [updated row provenance](../row-provenance.csv), [updated conservative group plan](../../signer-benchmark/identity-plan.csv).

The updated [source-group snapshot](source-group-plan.csv) constrains all 1,648 sessions across four families. [Historical replay enforcement](group-enforcement.json) confirms that the former split is rejected because three source families span partitions. The [signer-readiness audit](signer-readiness.json) still reports all 1,648 identities unverified; source constraints do not make that person-independent benchmark ready.
