> Updated recovery: [completed 89,774-image replay, 27,016 additional row links and source constraints for all 1,648 sessions](exhaustive/report.md). The bounded results below describe the earlier pass.

# Dataset source recovery — 24 October 2026

> Applied follow-up: [source grouping constraints for 45 sessions, with historical overlap enforcement](grouping.md). Source grouping is recorded separately from unverified signer identity.

## What was recovered

The investigation used the actual current CSV, import-session hash algorithm, original downloaded CSVs and ZIP archives, saved capture manifests, image re-extraction and existing fixed-benchmark predictions. Source recovery is work that can be automated locally; it does not require the owner to remember dataset names or identify people.

| Current rows | Recovered source | Evidence level |
|---:|---|---|
| 93,892 | `C:/Users/jibbi/Downloads/archive (4).zip` | 24 exact import-session hash links; all 93,892 rows also match `Downloads/landmarks (1).csv` in label, handedness and all 63 float32 coordinates |
| 1,800 | Three ZIP backups in `ml/data/manual_recordings/backups/` | Nine current session IDs appear in archived capture manifests, with 200 stored frames and 200 accepted rows per session |
| 63,842 | Legacy session families `ext_archive3_*` and `ext_archive5_*` | Family names plus sampled image evidence; not a complete row-by-row reconstruction |
| **159,534** | **Total current dataset** | Source dataset and model files unchanged |

The import/capture-level mapping covers **95,692 rows (59.98%) and 33 sessions**. It is recorded against the current dataset hash, not blindly copied from a stale report. The remaining legacy families contain 1,615 session strings; these include historical artificial row chunks and cannot be counted as people or independent visits.

Earlier project records already linked the 24 hashed import sessions to archive (4). The generic source-name grouping in the recent diagnosis did not apply those records and called that bucket unresolved. The current mapping corrects that incomplete interpretation. Sources and signer identities are different: recovering a ZIP does not identify each person inside it.

Artifacts:

- [Session-level provenance](session-provenance.csv)
- [Every current row and its source status](row-provenance.csv)
- [Downloaded CSV matches](csv-container-matches.json)
- [Manual capture-batch evidence](manual-batches.json)
- [Machine-readable summary and split coverage](summary.json)

## The source-related split problem is now concrete

The historical split assigns the entire archive-(4) source/label session to one partition. It puts that archive's W examples in validation, while putting its G, I, K, M, O, Q and X examples in test. Those source/letter combinations have **zero training rows from the same import container**. Training includes those letters from other sources; M additionally has the new local capture data. This does not mean a letter is entirely absent from training.

| Archive-(4) source/letter | Partition | Rows | Retained-model accuracy on these rows |
|---|---|---:|---:|
| W | Validation | 4,165 | 96.45% |
| G | Test | 4,126 | **2.23%** |
| I | Test | 3,940 | 92.94% |
| K | Test | 4,523 | **22.93%** |
| M | Test | 1,062 | 47.46% |
| O | Test | 4,146 | 53.91% |
| Q | Test | 3,627 | **29.00%** |
| X | Test | 3,894 | 47.15% |

The test rows in this table total 25,318, or 72.92% of the historical test. That is the same bucket previously scoring 41.15% overall accuracy. G/K/Q are immediate review priorities. This is source-conditioned **accuracy**, not per-letter F1 over the whole test.

This source/letter mismatch helps explain why an easy W-heavy validation set can favor changes that do not improve difficult test conditions. It is a measured coverage issue, not proof that every problematic image is mislabeled. Review sign convention, pose/viewpoint coverage and extraction quality before changing labels. The photographed source examples show different framing, viewpoints and backgrounds; those observations alone do not establish the cause of every error.

See [source/letter performance](source-label-performance.csv). The split, labels and evaluation rows were preserved.

## Image-level evidence from the remaining archives

A bounded probe re-extracted 120 source images using the current detector/import preprocessing. It reproduced the stored label, handedness and all 63 float32 coordinates within absolute tolerance 1e-6 for 41 current rows:

| Image archive | Matching current rows | Observed session family |
|---|---:|---|
| `archive (3).zip` | 9 | `ext_archive3_*` |
| `archive (5).zip` | 18 | `ext_archive5_*` |
| `archive.zip` | 14 | The hashed archive-(4) import sessions |

The 14 last rows are already covered by the import-level source mapping; they are not 14 additional independent examples. The 27 legacy rows provide individual image-replay evidence for the remaining families, but do not prove the origin of every other row in those families. A geometry match does not prove a unique original image, a correct linguistic label or a person's identity.

An interactive static [matched-image gallery](image-gallery.html) shows each recovered photo, dataset row, archive member, historical split and retained model prediction where available. The gallery requires no recollection of source filenames. [Exact probe records](image-match-evidence.json) include image SHA-256 values and coordinate differences.

## Some downloaded archives overlap

`archive.zip` has 87,028 image members. All 87,028 have CRC/size counterparts in `archive (4).zip`. Twenty candidate pairs were additionally read and verified **byte for byte**, with image SHA-256 recorded. The remaining candidate pairs have not all been cryptographically compared.

Therefore these ZIPs must not be treated as independent source datasets solely because their filenames differ. The archive-(4) package contains exact copies of images also distributed in archive.zip. This does not on its own prove current training/test row leakage: current row-level overlap must be checked separately, including transformed derivatives and differing extraction versions.

No CRC/size candidates were found between archive (4) and the local archive-(3)/(5) files. That does not rule out resized, re-encoded, rotated or otherwise transformed copies.

Evidence: [archive overlap and verified byte-equal examples](archive-overlap.json).

## Public dataset attribution: supported candidates, not guessed identities

The local archive (5) contains `SignAlphaSet/A/A_0.jpg`-style paths and 26,000 images. The publisher's [SignAlphaSet Version 1 record](https://data.mendeley.com/datasets/8fmvr9m98w/1) describes a 26,000-image A–Z collection. That is a strong name/size/structure match, but the exact downloaded version has not been established by comparing a publisher checksum. The record does not map each local file to a verified person.

The local archive (3) contains `dataset5/A/g/color_6_0002.png`-style paths, with five outer folders A–E, 65,774 color images and 65,894 depth images. The outer folders are **not alphabet classes**: the inner folders identify letters. The dataset structure is consistent with the [Pugeault/Bowden ASL Finger Spelling Dataset A](https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset), whose author describes RGB/depth data organized by subject, five users and 24 static letters. This suggests a path to recovering anonymous subject groups from original folders. Package identity and row-to-folder mappings still need complete verification before assigning those groups to all legacy rows.

These publisher links describe candidate origins; they do not verify that the local ZIPs are unchanged publisher downloads. No personal names, face recognition or inferred biometric identities were used. Anonymous IDs can be based on verified source subject metadata once the linkage is established; they should not be made distinct simply because sessions have different names.

## Changes applied

The existing signer-benchmark plan was backed up as [identity-plan.before-source-recovery.csv](identity-plan.before-source-recovery.csv). Its 33 verified source sessions now have `source_id`, `source_path` and `source_evidence` populated. No signer ID, global person group, split or identity-verification status was invented. [Enrichment receipt](applied-source-enrichment.json).

The person-separated readiness status remains false because the source container links do not establish all signer identities. That limitation does not prevent source-level review, image recovery or conservative source-family diagnostics. It is not a request for the owner to identify the sources manually.

## What remains and how to proceed

The remaining automated provenance work is to re-extract the color images from archive (3) and images from archive (5), then match them back to each current row, preserving ambiguous matches. This bounded probe already demonstrates that such matching can recover evidence. The local source archives are available, so it is not necessary to ask the owner to find their names. A full reconstruction is substantially larger than this 120-image probe and has not been completed here.

Source-based experiments must group identical originals and known derivatives together, and distinguish import-container separation from actual subject separation. A new benchmark should preserve the existing historical results and should not claim that renamed archives, synthetic chunks or invented people establish independence. Source/image review can now start with the specific files and rows linked above; corrections must be evidence-backed and separately recorded.

The current dataset SHA-256 is `03b949d3874b68bf0179c73fdfe918da7973ba89345e3b0d981171cc3da461d2`. The original dataset and all inspected application checkpoint hashes were verified unchanged. Nothing was retrained, relabeled or deleted during this investigation.
