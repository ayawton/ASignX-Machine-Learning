# ASignX - Hand and Moving-Sign Recognition

ASignX detects and tracks up to two hands with MediaPipe, exports each hand's
21 three-dimensional landmarks, and includes a video-to-LSTM pipeline for
training and recognizing moving signs and gestures.

The detector and classifier are deliberately separate: the detector can still
be used by itself, while sequence models are trained only for the labels in
your video dataset.

## Supported recognition workflows

- Landmark tracking: one or two hands at real-time webcam speed.
- Static pose classification: a single normalized landmark frame.
- Moving signs and gestures: a time-ordered landmark clip, including whole-hand
  travel and finger articulation. Typical moving classes include `J`, `Z`,
  `HELLO`, `THANK_YOU`, `PLEASE`, and `HELP`.
- Two-hand gestures: use the default `both` mode. Features are stored in stable
  `Left` then `Right` slots, with missed detections represented explicitly.

Recognition is data-driven. A label is supported after you provide and train
on representative videos for that label; the names above are examples, not a
built-in pretrained vocabulary.

## Installation

```bash
pip install -r requirements.txt
```

The main dependencies are MediaPipe Tasks, OpenCV, NumPy, PyTorch, and pytest.
The MediaPipe model is bundled at `asignx/assets/hand_landmarker.task`.

## Real-time landmark tracker

```bash
python main.py
```

Useful options:

```text
--camera CAMERA
--width WIDTH
--height HEIGHT
--max-hands {1,2}
--model-complexity {0,1}
--min-detection-confidence FLOAT
--exposure -5.0|-6.0|auto
--no-threads
--print-stream
```

Keyboard controls are `q`/Escape to exit, `d` to toggle landmark drawing, `h`
to toggle the HUD, `e` to cycle exposure, `p` to print landmarks, and `s` to
save a frame plus JSON.

## Train moving signs from videos

You can collect moving signs directly from the webcam, or import recordings.
Both methods save sequences for the same `ml.train_sequence` command.

### Standalone Windows GUI

Double-click **`dist/ASignX_DynamicStudio.exe`** (or `ASignX_DynamicStudio.bat`).
The single executable includes Python and the hand detector; recording PCs do not
need Python or PyTorch installed. The GUI provides webcam preview, a preparation
countdown, Start/Save/Discard controls, and labeled video folder extraction.
It saves the same `.npz` sequences used below; raw webcam video is not retained.
The portable EXE defaults to `data/sequences` and `data/video_sequences` beside
itself; choose the output folder explicitly when moving data back for training.

Source launch: `python -m ml.dynamic_studio_gui`. Build instructions and the full
workflow are in [Dynamic Sign Studio guide](docs/dynamic-studio-guide.md).
This GUI handles dynamic signs; `ASignX_ManualTraining.bat` opens the static studio.

### Option A: capture complete gestures from the webcam

Open a terminal in the project folder and run this once for each sign label:

```bash
python -m ml.collect_sequence --label J --takes 3
python -m ml.collect_sequence --label Z --takes 3
```

In the camera window, press `Space` to start a take, perform the entire moving
sign, then press `Space` again to save it. Repeat for the next take. `R`
discards the current take; `Q` or Escape exits. With `--takes 3`, the program
exits after three accepted takes. You can restart the same command later to
add more takes. Each saved take gets a unique session ID automatically.

The label comes from `--label`; there is no need to edit a Python file or CSV.
For a one-hand dataset, use `--hand Right`, `--hand Left`, or `--hand single`
consistently for every label. The default `--hand both` supports one- and
two-hand signs in stable Left/Right slots. Keep the same hand mode and
`--sequence-length` for every take in one output directory.

The collector samples at 15 FPS and saves each accepted gesture as a 30-step
`.npz` file under `ml/data/sequences/<label>/`. It rejects takes that are too
short, have too few hand detections, or lack usable landmarks. The default
maximum take length is 10 seconds. Change these settings with `--target-fps`,
`--sequence-length`, `--min-frames`, `--min-detection-ratio`, and
`--max-seconds` if needed.

Record at least three independent takes for each of at least two labels; more
varied takes will usually generalize better. Then follow "Train and evaluate"
below. Static capture with `ml.collect_data` writes `landmarks.csv` and is
trained separately with `ml.train`; it does not change these sequence files.

### Option B: import existing videos

#### 1. Arrange recordings by label

Each video is one independent recording session:

```text
videos/
|-- J/
|   |-- signer1_take1.mp4
|   |-- signer1_take2.mp4
|   `-- signer2_take1.mp4
|-- Z/
|   |-- signer1_take1.mp4
|   |-- signer1_take2.mp4
|   `-- signer2_take1.mp4
`-- HELLO/
    `-- ...
```

At least three videos per label are required so train, validation, and test
sets can use different sessions. Ten or more varied recordings per label is a
more realistic starting point. Do not cut one recording into copies across
these sets: that leaks nearly identical frames and produces misleading scores.

For each recording:

- Include the complete gesture, with a short neutral lead-in and lead-out.
- Keep camera orientation and mirroring consistent with live inference.
- Keep both hands visible for two-hand signs.
- Vary signer, distance, lighting, background, and speed across recordings.
- Use the same handedness policy for every class.

#### 2. Convert videos into landmark sequences

The default preserves both hands, samples at 15 FPS, and maps each complete
gesture to 30 timesteps:

```bash
python -m ml.prepare_videos --input videos --output ml/data/video_sequences
```

For a strictly one-hand dataset, select a hand consistently:

```bash
python -m ml.prepare_videos --input videos --output ml/data/video_sequences_right --hand Right
```

Important options:

```text
--sequence-length 30
--target-fps 15
--hand {single,Left,Right,both}
--min-detection-ratio 0.5
--overwrite
--skip-invalid
```

Preparation fails on malformed or low-detection videos by default. Each output
`.npz` stores the label, session ID, preprocessing contract, source metadata,
and a fixed-length feature sequence. Duplicate source videos are rejected.
Keep imported videos in a separate output directory from manually captured
takes, because rerunning the importer checks that its output matches its input.

Unlike static-pose preprocessing, sequence preprocessing uses one spatial
reference for the whole clip. It therefore retains wrist movement across time
instead of accidentally erasing the motion signal.

### Train and evaluate

```bash
python -m ml.train_sequence --data-dir ml/data/sequences --output ml/models/sequence_classifier.pth --epochs 50
```

Use `--data-dir ml/data/video_sequences` instead if you imported videos with
the command above. The sequence dataset must use one consistent hand mode and
sequence length.

Training creates:

- `sequence_classifier.pth`: model, labels, feature width, preprocessing
  version, sequence length, and training metadata.
- `sequence_classifier.splits.json`: deterministic, session-disjoint splits.
- `sequence_classifier.metrics.json`: validation/test metrics, per-class
  scores, and confusion matrix.

The held-out test set is evaluated only after selecting the best epoch by
validation loss. If dataset contents change, the saved split manifest is
rejected so an old split cannot silently be reused.

### Run live moving-sign recognition

```bash
python -m ml.predict_sequence --model ml/models/sequence_classifier.pth
```

For a one-hand checkpoint, pass the same selection used during preparation:

```bash
python -m ml.predict_sequence --model ml/models/sequence_classifier.pth --hand Right
```

Press `r` to clear the temporal window and `q`/Escape to exit. Predictions are
confidence-thresholded and stabilized across consecutive windows to reduce
duplicate output.

## Test static and dynamic recognition together

To test static and dynamic recognition in one camera window, double-click
`ASignX_TestRecognition.bat` or run:

```powershell
python main.py --sign-model ml/models/sign_classifier_canonical.pth --sequence-model ml/models/sequence_classifier.pth --exposure auto
```

Hold still signs for the static model and perform moving signs for the sequence
model; the existing motion gate selects the recognition path. Both trained
checkpoints are required. Press `q`/Escape to exit.

## Programmatic detector API

```python
from asignx import HandDetector, WebcamStream

with WebcamStream(camera_index=0, width=800, height=600) as camera:
    with HandDetector(max_num_hands=2) as detector:
        for frame in camera.stream():
            result, annotated = detector.process(frame, draw=True)
            if result.left_hand is not None:
                left_coordinates = result.left_hand.to_numpy()  # (21, 3)
```

## Landmark indices

MediaPipe's index order is preserved: wrist `0`; thumb `1-4`; index finger
`5-8`; middle finger `9-12`; ring finger `13-16`; and pinky `17-20`.

## F1 evaluation and improvement workflow

Source/session error review, validation macro-F1 selection, controlled ablations,
explicit global identity plans, threshold calibration and collection readiness are
documented in [F1 workflow](docs/f1-workflow.md). The current audit is at
`output/f1-review/baseline/review.html`. Unknown signer identities must be supplied
from real provenance; the tools never manufacture independent signers.

## Tests

```bash
pytest -v
```

The tests cover the detector, landmark models, motion-preserving sequence
normalization, missed-frame interpolation, webcam take controls, two-hand
ordering, sequence storage, checkpoint loading, and inference shapes.
