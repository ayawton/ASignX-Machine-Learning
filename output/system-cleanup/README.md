# Extractor and build cleanup

Kept the reviewed extractor at `output/extractor-reviewed/ASignX_DataExtractor` and its matching distribution ZIP. Before removal, all 2,016 package files were compared with the ZIP by SHA-256. The packaged executable passed a three-image import smoke check, correctly skipping two conflicting-label images and finding no hand in the blank example. This verifies package startup and this import path, not extraction accuracy on every image.

Removed the older extractor packages from `output/extractor`, `output/extractor-large-datasets`, and `dist/ASignX_DataExtractor`; build intermediates from `build` and five temporary build directories; and source/test caches. Total removed: approximately 2.37 GiB. Exact target paths, sizes, and protected-file hashes are in `cleanup-record.json`.

The extractor launcher now selects the retained package, falling back to Python source when unavailable. The guide describes this behavior. Build outputs and the actual Python environment directory were added to `.gitignore`. The Dynamic Studio executable, both application model checkpoints, current dataset, historical dataset backups, correction evidence, temporary audit/recovery scripts, and temporary dependency installations were preserved.

Protected-file hashes matched after cleanup. Previous launcher, guide, and ignore settings are saved under `config-before`. Git's index was not modified; tracked cache-file removals remain working-tree changes for the next commit.
