# Static dataset importer

ZIP imports have no entry-count limit. The total uncompressed-size limit remains 10 GiB, and archive path checks remain enabled.

Identical image files or hand-point vectors assigned to different signs are skipped and recorded in the detailed results, while other images continue importing. Every conflicting copy in the new import is skipped; the importer does not guess which label is correct or alter existing CSV rows. Review those source paths and correct their labels before importing them again.

Run `ASignX_DataExtractor.bat` on Windows, or `python -m ml.extractor_gui` in a compatible Python environment. The launcher prefers `output/extractor-reviewed`, then `output/extractor-large-datasets`, then `output/extractor` when present; it does not launch the older binary in `dist`.

1. Choose a dataset folder or ZIP and the destination `landmarks.csv`.
2. Select **Preview sign names**. Common layouts such as `train/a/*.jpg`, `A/*.png`, and `sign_hello/*.jpg` are inferred automatically. `A` and `a` become `A`. Folder guessing cannot resolve every dataset: check the preview.
3. For an unusual layout or known signers, supply a **Sign names file**. It is a CSV with each image's relative path, the intended label, and an optional real origin group. Every discovered image must be listed. Paths inside ZIP files include any top-level wrapper folder.

```csv
path,label,group
images/person_1/001.jpg,A,person_1
images/person_2/002.jpg,A,person_2
images/person_3/003.jpg,A,person_3
```

4. Choose a cleaning level:

   - **Keep varied examples (recommended):** remove unreadable images, unusable points, exact file repeats and exact landmark repeats. Keep pose diversity.
   - **Remove very similar examples:** also reduce almost identical hand poses.
   - **Also remove unusual poses:** also remove examples far from the usual pose of that sign. This can remove legitimate variations; inspect the results before training.

5. Optionally select a separate folder for accepted image copies. Source images stay untouched. Rejected images are not copied.
6. Select **Extract and add new data**. Review counts, warnings and the JSON results file saved beside the CSV. It records the relative source path, label, file hash when available, origin group and decision for every discovered image.

The importer accepts JPG, JPEG, PNG, BMP and WebP still images. It rejects datasets containing known video formats or GIFs. Animated PNG/WebP files are checked during image reading. It cannot determine whether a photographed sign actually requires movement. Verify the label definitions in your target sign language. Images with more than one detected hand are skipped because the static training format stores only one hand.

Existing CSV rows must pass the shared schema, preprocessing and numerical checks. They are preserved byte for byte when adding data. Existing split manifests and models are retained as evidence; training detects a changed dataset hash and creates a new split. Keep a separate frozen test dataset for meaningful model comparisons.

An extraction does not need manual train/validation/test folders. Training still needs at least three independent origin groups per sign. Randomly dividing images from one source does not create independence. When origin is unknown, images from that source and sign stay together and a warning explains why training may need more data. Use a stable **Dataset source name** when importing copies of the same dataset from different paths/devices. Do not assign different group names to frames or augmentations from the same recording.

The session-based trainer does not enforce signer separation across different signs, nor does it preserve a dataset's published train/test partition. For a signer-independent benchmark, keep a separate signer-level manifest spanning every class and use a grouped evaluator. The report explains this remaining training change.

CLI example:

```powershell
py -3.11 -m ml.extract_dataset --input C:\datasets\static.zip --output ml/data/landmarks.csv --manifest C:\datasets\labels.csv --source-id corpus-2026 --cleaning gentle
```

For the optional Windows build:

```powershell
py -3.11 -m PyInstaller --noconfirm --distpath output/extractor-reviewed --workpath tmp/extractor-build ASignX_DataExtractor.spec
```

Copy the entire `ASignX_DataExtractor` output folder to another compatible Windows device, including `_internal`. A Windows executable does not run on Android, iOS, macOS or Linux. Those require a compatible Python installation, a build for that operating system, or the future browser extractor described in the report.

Imports use a cooperative `.lock` file and a changed-file check before atomic replacement. Stop other tools that write the same CSV. A crashed process can leave a lock; remove it only after verifying that no extraction is running. External writers such as camera collection do not yet participate in this lock. Atomic replacement protects the destination from partial writes, but there is a narrow external-writer race between the last check and replacement.

Cleaning applies to new samples only. It cannot establish correct sign labels or retroactively repair old provenance. Use the audit to review unusual examples and collect difficult cases; do not delete rare but correct hand shapes just to improve a benchmark score.
