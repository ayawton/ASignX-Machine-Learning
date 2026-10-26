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

## Body context for wider ASL signs

Launch `ASignX_DynamicStudio.bat` for the updated build in
`dist/tracking/ASignX_DynamicStudio.exe`, or run `python -m ml.dynamic_studio_gui`.

Choose **Landmarks: hands + body** with **Hand mode: both**. Use a fresh output
folder, such as `ml/data/sequences_body`, consistently across all labels. Keep one
signer in view, including their face, both shoulders, and arms.
The preview marks nose, chin, shoulders, and elbows. Face
tracking supplies the chin; pose tracking supplies the other reference points.
The trackers use the official MediaPipe [pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
and [face](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) Tasks APIs.

This mode saves 150 features per frame: two 63-feature handshape blocks, followed
by body and wrist positions relative to the shoulder midpoint, normalized by
shoulder width. Body context uses image-plane x/y plus visibility masks; it does
not mix incompatible pose/face/hand depth origins. Finger depth remains in each
handshape block. Missing references are masked, and takes with insufficient
shoulder visibility are rejected. The coordinates preserve hand travel relative
to the body while handshape remains normalized around each wrist.

This mode works for any dynamic label, including J, Z, HELLO, and THANK_YOU;
it is not restricted to words. Use one landmark mode across a model's dataset.
The hands-only option remains available in the same Studio.

The previous 156-feature format included hips. It is incompatible with this
150-feature version; re-extract original videos or record fresh data and train
a new checkpoint.

Original hand-only `.npz` files cannot supply these extra landmarks. Re-record or
re-extract original videos. Do not pad them using `unify_sequence_width.py` or mix
hand-only files into a body-context dataset. The legacy hand-only 3D augmentation
is disabled for this format so visibility masks and body coordinates stay valid.

```powershell
python -m ml.collect_sequence --label HELLO --hand both --body-context --output ml/data/sequences_body --exposure auto
python -m ml.prepare_videos --input videos --output ml/data/sequences_body_import --hand both --body-context
python -m ml.train_sequence --data-dir ml/data/sequences_body --output ml/models/sequence_body_v1.pth --epochs 50
python main.py --sign-model ml/models/sign_classifier_canonical.pth --sequence-model ml/models/sequence_body_v1.pth --exposure auto
```

Collect at least two labels with three independent sessions per label; substantially
more varied data is needed for useful recognition. Live recognition automatically
selects body tracking for a body-context checkpoint. Existing hand-only checkpoints
continue using hand tracking. Added pose/face detection costs processing time; camera
speed and recognition quality need testing on the target computer. Body references
provide location context, not ASL grammar or dedicated J/Z handshape validation.

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
Live sequence prediction uses automatic camera exposure by default. To set it manually,
add e.g. `--exposure -5.0`; use `--exposure auto` to restore automatic brightness.

## Test static and dynamic signs together

Double-click `ASignX_TestRecognition.bat` in the project folder, or run:

```powershell
python main.py --sign-model ml/models/sign_classifier_canonical.pth --sequence-model ml/models/sequence_classifier.pth --exposure auto
```

Both models share one camera window. Hold a still sign for static recognition;
perform a moving gesture for dynamic recognition. The motion gate routes recognition
based on hand movement, and the overlay shows the results. Only labels learned by
the respective models can be recognized. Press **Q** or **Esc** to exit.

Combined recognition captures a moving gesture with 200 ms of preceding landmarks.
Static confirmation is blocked during movement and until 400 ms of quiet has elapsed.
The completed take is sampled at the checkpoint FPS and resampled to its training
sequence length before classification. Static recognition then requires a fresh
stable pose. The dynamic result stays visible for 600 ms and emits once per take.
Long tracking gaps reject a take; a camera stall clears it. Continuous movement
beyond 10 seconds aborts the take and requires quiet before another can begin.
Motion uses the fastest tracked wrist/fingertip point and a lower exit threshold
to reduce switching around the motion threshold. Small tracking jitter can therefore
require adjusting `--motion-threshold` on the recognition command.

The launcher uses the project Python environment and requires both trained checkpoints.
It is separate from the portable recording/extraction EXE, which does not run recognition.
For different model versions, use the command above with their checkpoint paths.

## Hand tracking during fast movement

Hand detection runs on every processed frame. The default palm detection threshold
is 0.5 and hand-presence threshold is independently set to 0.45; tracking IoU remains
0.5. These are separate controls in the [MediaPipe Tasks API](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).
The duplicate filter compares corresponding landmark geometry relative to hand size
so touching wrists or overlapping hand boxes alone do not remove a hand.

When a recently tracked hand disappears, a full-image hand detector retries the
current frame at most once per 150 ms, for up to 500 ms since the last complete
detection. Recovery adds inference work only during recent tracking loss. It does
not reuse or predict old hand landmarks. Pose and face tracking refresh at up to
20 FPS, with body references cached for less than 50 ms between refreshes; hand
tracking continues on every frame. This prioritizes hand processing when using
body-context models. Static-image extraction still detects all references per image.

Test recognition accepts `--min-detection-confidence`, `--min-presence-confidence`,
and `--min-tracking-confidence`. Add `--no-hand-recovery` to compare without the
extra recovery pass. For example:

```powershell
.\ASignX_TestRecognition.bat --min-detection-confidence 0.5 --min-presence-confidence 0.45
```

Flat or edge-on poses and motion blur still need testing on the actual camera.
Bright, even lighting helps permit a shorter exposure. You can compare e.g.
`--exposure -6` with auto-exposure, but manual values are camera-dependent and may
make the image too dark. Added sign-training data does not retrain MediaPipe's
hand detector. No sign-model retraining is required for these tracking changes.

## Run from source or rebuild

```powershell
python -m pip install -r requirements-dynamic-build.txt
python -m ml.dynamic_studio_gui
.\build_dynamic_studio.ps1 -Python python
```

Build on Windows x64 with Python 3.14. The build produces `dist/tracking/ASignX_DynamicStudio.exe`
and a copy of this guide. The single EXE includes its runtime and hand model; no separate
`_internal` folder is needed. The existing static training studio and static extractor
remain available through their original launchers.
