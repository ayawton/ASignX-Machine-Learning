# ASignX Dynamic Sign Studio

Launch `ASignX_DynamicStudio.exe` on a Windows x64 PC. Python, PyTorch, and an internet
connection are not needed on the recording PC. The hand detection model is bundled.
The executable extracts its runtime to a temporary folder at startup, so the first
launch may take a little while. Keep the EXE in a writable folder, or choose a writable
output folder in the app.

## Record a moving sign

1. Choose **Webcam recording** and an output folder.
2. Enter a label such as `J`, `Z`, or `HELLO`. Choose a camera index (`0` is normally
   the built-in camera) and hand mode. `both` supports one- and two-hand gestures.
3. Click **Open camera**. Keep the entire gesture and required hands in view.
4. Click **Start take**. Wait for the three-second countdown, then perform the
   complete gesture. Click **Save take** when finished. A take automatically finishes
   after the maximum duration (10 seconds by default).
5. Click **Discard take** to cancel a countdown or active take. Rejected takes are
   not saved; start again. Closing the camera discards an unfinished take and keeps
   previously saved takes.
6. Close the camera to change the label/settings, then record the next sign.

Defaults sample at 15 FPS and resample the complete take to 30 sequence steps.
At least 10 sampled frames and a hand visible in at least 50% of samples are required.
The detection ratio counts frames with any selected hand; it does not prove that both
hands were visible or that the sign was correct. Check the preview while recording.
The app retains whole-hand travel along with finger articulation.

Each accepted take creates `<output>/<label>/<unique-session>.npz`. These files
contain extracted landmarks and metadata, **not original camera footage**. Record
original clips separately if you need to review or re-extract them later.

Use the same hand mode and sequence length across all labels in a dataset. An output
folder containing incompatible sequences is rejected. Collect at least three independent
takes per label and at least two labels for training. More varied takes are preferable;
vary signer, lighting, speed, position, and recording conditions.

## Extract existing video recordings

Arrange one complete sign per video:

```text
videos/
  J/
    take1.mp4
    take2.mp4
    take3.mp4
  Z/
    take1.mp4
    take2.mp4
    take3.mp4
```

In **Extract from videos**, select the `videos` root and a separate output folder.
The importer uses hand mode, sequence steps, sampling FPS, and minimum detection ratio
from the recording tab. Click **Extract videos**. Supported extensions: `.mp4`, `.avi`,
`.mov`, `.mkv`, `.m4v`, `.webm` (decoding depends on the codec).

The importer requires at least two labels and three valid unique videos per label.
It rejects copied videos and low-detection recordings. With **Skip invalid videos**,
bad recordings can be skipped, but the minimum valid-video count still applies.
Errors or stopping can leave completed sequences in the output; matching files are
reused when rerunning the same import. Stop/close waits until the current video
finishes. Existing files that mismatch the source/settings, or output files absent
from the selected input dataset, cause an error. Use a fresh output folder in that case.
Do not mix imported files and webcam takes in one output folder.

## Train in the project

This executable collects/extracts data. Run training in the Python project with PyTorch:

```powershell
python -m ml.train_sequence --data-dir "PATH_TO_SEQUENCES" --output ml/models/sequence_classifier.pth --epochs 50
python -m ml.predict_sequence --model ml/models/sequence_classifier.pth
```

For a one-hand model, pass the same hand selection to live prediction, e.g. `--hand Right`.
Copied sequence files retain their independent recording session IDs.

## Run from source or rebuild

```powershell
python -m pip install -r requirements-dynamic-build.txt
python -m ml.dynamic_studio_gui
.\build_dynamic_studio.ps1 -Python python
```

Build on Windows x64 with Python 3.14. The build produces `dist/ASignX_DynamicStudio.exe`
and a copy of this guide. The single EXE includes its runtime and hand model; no separate
`_internal` folder is needed. The existing static training studio and static extractor
remain available through their original launchers.
