> Approved cleanup completed: [actual deletions, preserved system files and tested restoration instructions](completed-cleanup.md). One nonduplicate distribution ZIP was kept.

# Further cleanup recommendations ? 24 October 2026

Audited workspace file sizes, application launcher dependencies, dataset hashes, ZIP-contained CSV hashes and derived-sample membership. The initial audit was read-only; the approved cleanup is now recorded separately. Values below use MiB (1,048,576 bytes).

## Recommended first pass

| Category | Recoverable size | What to remove |
|---|---:|---|
| Duplicate current CSV | 392.08 MiB | Two dataset copies under the old manual run and full-static experiment; preserve their manifests and restore mapping. |
| Duplicate interim backups | 389.85 MiB | Two loose CSV backups already preserved byte-for-byte inside retained capture ZIPs. |
| Distribution ZIP copies | 142.80 MiB | Extractor and CSV-merger distribution ZIPs; retain their extracted application folders. |
| Temporary artifacts | 116.23 MiB | Temporary parquet dependencies, generated ZIP-limit test files and PDF preview images/text/logs; retain utility/builder scripts. |

**Total recommended first pass: 1.02 GiB**, before small Python/pytest caches. The [exact candidate list](cleanup-candidates.json) records each file, its reason and restoration source where applicable. Duplicate CSV removal does not lose data; experiment paths and restore instructions should be updated as part of cleanup.

## Optional second pass

- **346.47 MiB of derived experiment snapshots:** the added-M candidate CSV and source-separated candidate CSV contain no label/hand/float32 geometry absent from the main dataset. Preserve exact original ordering, session metadata, pinned master version, row ledgers and verified reconstruction recipes before deleting. Geometry membership alone does not prove byte-for-byte reconstruction.
- **Potential 1.4 GB of local Git LFS cache:** `git lfs prune --dry-run` identified 223 prunable objects, with 116 retained. Remote availability was not verified. Use supported Git LFS cleanup after that check; do not manually remove `.git` objects or rewrite history for routine cleanup.
- **115.29 MiB historical legacy ZIP:** runtime does not need it, but it holds unique historical datasets, a legacy model and merge evidence. Keeping or externally archiving it preserves old experiments; it is not part of the recommended first pass.

## Preserve for the whole system

- `asignx/`, `ml/` source, `main.py`, `web_client/`, launchers, build specifications, requirements, setup scripts and tests.
- `.venv-py314/` (about 1.14 GiB): current recognition/training runtime dependencies.
- `ml/data/landmarks.csv`, sequence recordings and the two active recognition checkpoints with metadata/split manifests.
- All three capture ZIPs in `ml/data/manual_recordings/backups/` (791.93 MiB): they contain **1,800 raw M frames**, manifests and frozen CSV versions. Those frame images are not present as unpacked recordings in this workspace. Do not delete them as generic backups.
- `ml/data/landmarks.csv.backup-20261022T062606Z-b665b450a219`: the earliest reviewed CSV snapshot is not one of the byte-identical redundant interim copies.
- `output/datasets/weak-classes/`: its extracted CSV contains **4,118 rows absent from the main CSV**. These remain data for review; no assertion is made that their labels are correct. The old combined weak-class dataset has 6,371 geometries absent from main, so it should not be discarded as an exact main-dataset duplicate.
- `archive/data-corrections/`: original data versions and review evidence, including the exact historical dataset identified by the active checkpoint.
- `output/source-recovery/`, `output/signer-benchmark/`, split manifests, prediction CSVs, comparison reports and cleanup logs: retain provenance and evaluation evidence.
- `output/extractor-reviewed/ASignX_DataExtractor/`, `output/csv-merger/ASignX_CSVMerger/`, `dist/ASignX_DynamicStudio.exe`, and current ONNX exports. The batch launchers directly reference these locations. Deleting all of `output/` or `dist/` is not a system-preserving cleanup.

## Evidence

[Folder/file sizes](inventory.json), [dataset SHA-256 comparisons and capture contents](datasets-and-backups.json), [hashes inside capture ZIPs](archive-csv-hashes.json), [derived-sample membership audit](derived-data-audit.json).
