> Updated constraints: [the exhaustive pass now supplies source co-location evidence for all 1,648 sessions](exhaustive/report.md). The 45-session application below is the earlier bounded pass; signer identities remain unverified.

# Recovered source grouping constraints — 24 October 2026

Applied the recovered source relationships to the benchmark identity plan as **co-location constraints**, not claims of independent people.

## Added

- **45 constrained sessions across four source families.** The 33 import/capture sessions have whole-session source records. Another 12 legacy sessions have partial image-replay evidence: the whole session is kept with the matched source conservatively, without claiming every row exclusively originated there.
- `archive.zip` and `archive (4).zip` share one family because exact duplicate image bytes connect them.
- The nine local M sessions are grouped into one capture collection. Three batches do not establish three different signers.
- Sample-matched legacy sessions are linked to the corresponding archive-(3) or archive-(5) family. Unsampled legacy chunks are not automatically attributed by their names.
- Added `group_kind` and `group_evidence` so source constraints cannot be mistaken for verified signer identities.

The updated plan is [source-group-plan.csv](source-group-plan.csv), also applied to `output/signer-benchmark/identity-plan.csv`. The previous plan is preserved as [identity-plan.before-source-grouping.csv](identity-plan.before-source-grouping.csv).

## Verification

The existing group-plan validator rejects a replay of the historical partition assignments because related source families cross partitions. Three of the four recovered families span multiple historical partitions. [Audit and enforcement result](source-group-audit.json); [historical replay plan](historical-source-group-replay.csv).

This is the intended constraint: when a new split is constructed, related source members must stay together. It does not imply that the historical model scores have been recalculated on an independent benchmark.

## Remaining limits

**1,603 sessions remain without a recovered source-family constraint.** No signer IDs, verified-person status or new split assignments were invented. The signer-readiness assessment remains false. Co-locating known sources prevents specific known overlaps; it cannot certify independence between all remaining samples.

No dataset rows, existing split assignments or model weights were changed. A complete new benchmark must resolve or explicitly exclude unknown material and still provide all labels in each partition. Source-separated and signer-separated results must be described separately.
