# Completed safe storage cleanup ? 24 October 2026

Deleted **989 audited files**, removing **1,067,434,515 bytes** (0.99 GiB; 1.07 GB) from the working tree. Deletion was limited to exact regular-file paths in the verified execution manifest; no recursive directory deletion was used.

Removed two byte-identical main-dataset copies, two interim CSV backups already preserved in capture ZIPs, the matching extractor distribution ZIP, old temporary parquet dependencies, generated ZIP-limit test artifacts and PDF preview images/text/logs. Utility and PDF builder scripts were retained.

The CSV-merger distribution ZIP was **kept**: its contents differ from the retained Python 3.14 application package. It is an older build, not a verified identical distribution copy. The full comparison is saved in the execution plan.

## System and data verification

- Checked 5,975 protected files before and after cleanup. Their hashes were unchanged at completion, including active recognition models, training/sequence data, capture ZIPs, source code, packaged applications and provenance records.
- Loaded both active checkpoints and successfully ran finite-output inference: static 24-label output and dynamic J/Z output.
- Imported detector/camera modules and all four source GUIs successfully. No camera or visible GUI was opened.
- Restored the deleted full-static dataset snapshot from its pinned capture ZIP, verified its exact SHA-256 and size, then removed that tested duplicate again.
- Preserved the three raw-capture ZIPs, earliest reviewed CSV backup, weak-class extracts, conditional experiment datasets, Python environment, Git history and Git LFS cache.
- Updated the current extractor guide to describe the removed distribution ZIP; source/runtime behavior was not changed.

## Restore a historical dataset copy

The four removed CSV copies remain byte-for-byte recoverable from retained capture ZIPs. List them with:

```powershell
.\.venv-py314\Scripts\python.exe output/storage-audit/20261024/restore_snapshot.py --list
```

For example, restore the dataset path expected by the old full-static experiment:

```powershell
.\.venv-py314\Scripts\python.exe output/storage-audit/20261024/restore_snapshot.py output/full-static-retraining/20261023T195828Z/dataset.csv
```

Restoration verifies both the source archive and extracted CSV hashes, and refuses to overwrite an existing path. Keep the original split manifests and model/evaluation records with the restored dataset.

To recreate an extractor distribution ZIP from the retained package when needed:

```powershell
Compress-Archive -LiteralPath output/extractor-reviewed/ASignX_DataExtractor -DestinationPath output/extractor-reviewed/ASignX_DataExtractor.zip
```

[Exact verified deletion/protection manifest](execution-plan.json), [cleanup result](cleanup-result.json), [restoration recipes and hashes](snapshot-restoration.json), [restoration tool](restore_snapshot.py), [inference/import smoke checks](smoke-check.json).
