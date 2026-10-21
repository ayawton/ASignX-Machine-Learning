# ASignX CSV Merger

The merger is a separate app from the extractor. It combines compatible static
landmark CSVs without modifying existing main rows or the incoming file.

## Open and merge

1. Run `ASignX_CSVMerger.bat` in the project, or open the portable
   `output/csv-merger/ASignX_CSVMerger/ASignX_CSVMerger.exe`.
2. Choose **Main CSV**: select this project's `ml/data/landmarks.csv`.
   This file takes priority and receives the new rows. Check the path carefully.
3. Choose **Incoming CSV**: the CSV transferred from the other device.
4. Select **Preview merge** and review totals and skipped rows.
5. Select **Merge into main CSV** and confirm. Backup and JSON report paths
   appear on completion. Both files are saved beside the main CSV.

Close extraction, collection, training, and other CSV writers before merging.
The merger uses the extractor's `.lock` naming protocol and checks both file
hashes before saving. Collection and external editors may not respect this lock;
there remains a small race between the final check and replacement. A stale lock
must only be removed after verifying that no writer is running.

## Merge rules

- Both headers must match the current 67-column format and preprocessing version.
  Incompatible files stop the operation; the app does not convert legacy CSVs.
- Main rows must be valid and are preserved as original bytes. Existing main
  duplicates or conflicting poses are retained; this app does not repair them.
- Malformed incoming rows are skipped and reported. Features use the training
  validator: 63 finite values, wrist at the origin, maximum hand radius one.
- Exact landmark equality uses the same float32 precision as training, across
  all labels and handedness. Signed zero is equivalent. Similar poses are kept.
- A matching main sample wins. Incoming matching vectors with different labels
  are skipped. If new incoming vectors have conflicting labels, all those copies
  are skipped instead of choosing a label by file order.
- Source/session IDs and labels are preserved. Sessions assigned to different
  labels are skipped, not renamed. Valid incoming duplicates keep the first copy.
- No samples are appended until confirmation. Changed inputs require a new
  preview. Reimporting the same file does not append the same landmarks again.

Use the same dataset source name during extraction on both devices when they
contain the same source data. Different session IDs do not prove independence.
CSV files contain no image hashes, so this app cannot detect the same original
image if its extracted landmarks differ. It cannot verify sign labels or recover
missing origin information. Do not invent source IDs to bypass conflicts.

## Backup and report

Each committed merge saves `landmarks.csv.backup-<UTC timestamp>-<unique ID>`
and `landmarks.csv.merge-<UTC timestamp>-<unique ID>.json`. To restore, close
writers and copy the backup over the main CSV. Preserve the backup separately.
Reports include input hashes, the expected merged hash, counts and every incoming
row's decision. Reports are written before replacement; if saving fails, compare
the main file's SHA-256 with `expected_output_sha256` to confirm whether it applied.
Existing trained models and split files are not changed; retrain for new data.

## Transfer to another Windows computer

Copy the entire `output/csv-merger/ASignX_CSVMerger` folder, including `_internal`.
ZIP that folder for USB or cloud transfer and extract it before launching the EXE.
The portable app includes Python and NumPy. Transfer datasets separately. A Windows
build requires compatible Windows architecture; it does not run on phones or macOS.

## Development

Run from source with `py -3.11 -m ml.csv_merger_gui` (NumPy and Tkinter required).
Build with Python 3.11, NumPy and PyInstaller installed:

```powershell
py -3.11 -m PyInstaller --noconfirm --distpath output/csv-merger --workpath tmp/csv-merger-build ASignX_CSVMerger.spec
```

Run the focused tests with `py -3.11 -m pytest tests/test_csv_merger.py`.
