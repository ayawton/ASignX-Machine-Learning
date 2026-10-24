# Python 3.14 project environment

All active Python workflows use standard Windows x64 Python 3.14.7 in
`.venv-py314`: recognition, training, collection, extraction, CSV merging,
evaluation, reporting, ONNX export and Windows packaging. The browser SDK
continues to run JavaScript/TypeScript.

## Setup and commands

```powershell
.\setup_python314.ps1
.\.venv-py314\Scripts\Activate.ps1
python --version
python -m pip check
python -m pytest -q
python docs/build_detailed_system_report.py
```

Alternatively invoke `.\.venv-py314\Scripts\python.exe` directly. The batch
launchers prefer the local environment for source execution and use `py -3.14`
if it is absent. Install the environment first for reproducible dependencies.
Packaged tools remain executable-first and now bundle Python 3.14.

`requirements.txt` covers recognition/training. `requirements-tools.txt` adds
the tested reporting, export and packaging versions. The complete installed
Windows environment is recorded in `requirements-py314.lock.txt`; install it
with `python -m pip install -r requirements-py314.lock.txt` to reproduce those
versions. Version pins do not replace clean-machine testing.

## Build portable Windows tools

With the project environment activated:

```powershell
python -m PyInstaller --noconfirm --distpath output/extractor-reviewed --workpath build/extractor ASignX_DataExtractor.spec
python -m PyInstaller --noconfirm --distpath output/csv-merger --workpath build/csv-merger ASignX_CSVMerger.spec
.\build_dynamic_studio.ps1
```

Distribute the whole extractor or merger directory, including `_internal`.
Dynamic Studio is a single executable. Recreate the extractor distribution ZIP
after rebuilding so that it matches the package. Source changes do not update
an executable automatically. Reports from before this migration describe their
historical environment and metrics; their old Python 3.11 references are not
current setup instructions.

## Verified on 22 October 2026

- Python 3.14.7, PyTorch 2.14.0, MediaPipe 1.0.1, OpenCV 5.0.0.93,
  NumPy 2.5.3; no broken requirements reported by pip.
- Full regression suite: 114 passed, seven export-related warnings in 29.95s.
  CSV Merger focused checks after its hidden self-test addition: ten passed.
- Both static checkpoints were compared on 128 fixed samples and the dynamic
  checkpoint on one stored sequence: Python 3.11/3.14 logits and predictions
  matched exactly. This is a numerical migration check, not a new accuracy study.
- All three real checkpoints exported to ONNX and ran in ONNX Runtime CPU.
  Maximum absolute output difference from PyTorch was below 0.000004.
- One original image passed the extractor retry path in both environments:
  one Right hand in each; maximum raw landmark difference about 0.000365.
  This does not establish equal extraction over all images or live cameras.
- Rebuilt extractor imported the copied image into a temporary CSV. Rebuilt
  Dynamic Studio passed its hidden Tk/MediaPipe/sequence-storage self-test;
  rebuilt CSV Merger passed its hidden Tk/NumPy startup self-test.
- The latest detailed PDF was generated using dependencies installed in the
  Python 3.14 environment and all rendered pages were inspected.
- The real dataset and all three trained checkpoints retain their hashes.

ONNX export now disables verbose progress output, avoiding Unicode progress
glyph failures on redirected Windows cp1252 streams. The exporter still has
its existing TorchScript fallback; verification explicitly requires an actual
ONNX artifact rather than accepting that fallback as browser compatibility.

Evidence, output comparisons, build/test logs, package hashes and storage
accounting are saved under `output/python314-migration/`.

The rebuilds were checked on this development machine, without a live camera
session, full linguistic audit, browser benchmark or separate clean Windows PC.
Those limits remain. No production retraining or model promotion was performed.

## Storage and Python 3.11

The active three applications plus extractor distribution ZIP decreased from
1,105,848,173 to 617,801,083 bytes, a reduction of about 465 MiB. This comparison
is for packaged deliverables, not total machine storage: additional tooling was
installed in the 3.14 environment. Superseded packages, temporary build outputs
and about 82 MiB of redundant temporary report dependencies were removed.

The migration replaces older project packages, rebuild intermediates and
temporary dependency directories after verification. It preserves datasets,
dataset backups, trained models, audit records and the active 3.14 environment.
System-wide Python 3.11 and its packages are retained because other projects
may depend on them. This project no longer requires that interpreter for its
active workflows. A browser runtime still does not require a Python installation.
