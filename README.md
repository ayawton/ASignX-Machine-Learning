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

### 1. Arrange recordings by label

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

### 2. Convert videos into landmark sequences

The default preserves both hands, samples at 15 FPS, and maps each complete
gesture to 30 timesteps:

```bash
python -m ml.prepare_videos --input videos --output ml/data/sequences
```

For a strictly one-hand dataset, select a hand consistently:

```bash
python -m ml.prepare_videos --input videos --output ml/data/sequences_right --hand Right
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

Unlike static-pose preprocessing, sequence preprocessing uses one spatial
reference for the whole clip. It therefore retains wrist movement across time
instead of accidentally erasing the motion signal.

### 3. Train and evaluate

```bash
python -m ml.train_sequence --data-dir ml/data/sequences --output ml/models/sequence_classifier.pth --epochs 50
```

Training creates:

- `sequence_classifier.pth`: model, labels, feature width, preprocessing
  version, sequence length, and training metadata.
- `sequence_classifier.splits.json`: deterministic, session-disjoint splits.
- `sequence_classifier.metrics.json`: validation/test metrics, per-class
  scores, and confusion matrix.

The held-out test set is evaluated only after selecting the best epoch by
validation loss. If dataset contents change, the saved split manifest is
rejected so an old split cannot silently be reused.

### 4. Run live moving-sign recognition

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

## Programmatic detector API

```python
from asignx import HandDetector, WebcamStream

with WebcamStream(camera_index=0, width=640, height=480) as camera:
    with HandDetector(max_num_hands=2) as detector:
        for frame in camera.stream():
            result, annotated = detector.process(frame, draw=True)
            if result.left_hand is not None:
                left_coordinates = result.left_hand.to_numpy()  # (21, 3)
```

## Landmark indices

MediaPipe's index order is preserved: wrist `0`; thumb `1-4`; index finger
`5-8`; middle finger `9-12`; ring finger `13-16`; and pinky `17-20`.

## Tests

```bash
pytest -v
```

The tests cover the detector, landmark models, motion-preserving sequence
normalization, missed-frame interpolation, two-hand ordering, sequence storage,
checkpoint loading, and inference shapes.
