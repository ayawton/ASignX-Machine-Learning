# ASignX Machine Learning Training, Extraction & Deployment Guide

A comprehensive, easy-to-follow guide on how to extract and clean datasets (via the portable `.exe` GUI or CLI), collect live data, train the upgraded **SignResMLP** and **TCN-Attention** models, run unified static + dynamic inference, and export models for client-side web deployment (**ASignLearn** & **ASignLive**).

---

## Which Model Are You Training?

ASignX provides a **Dual-Path Unified Architecture**:

1. **Static Pose Model (`ml.train` — Kinematic `SignResMLP`)**: For still signs (e.g., alphabet letters `A`–`Y`, numbers, static poses). Automatically enriches 63D normalized landmarks into **91D kinematic features** (joint flexion cosines, pairwise fingertip distances, and 3D palm normal vector) and uses a **Cosine Prototype Head** supporting instant zero-retrain sign registration.
2. **Moving Sign / Dynamic Model (`ml.train_sequence` — `TCN-Attention` / `SignLSTM`)**: For gestures involving motion (`J`, `Z`, `HELLO`, `THANK_YOU`). Uses dilated 1D Temporal Convolutions + Multi-Head Self-Attention across 30-frame `.npz` clips.
3. **Unified Motion-Gated Router (`main.py` / `ml.predict`)**: Automatically tracks hand velocity in real time to route still holds to `SignResMLP` and active trajectories to `TCN-Attention` while suppressing transition false positives.

---

## Part 1: Static Hand Signs (Letters A–Y, Static Poses)

### Option A: Portable Standalone Executable & Desktop GUI (Recommended for Any Device)
You can copy the **`dist/ASignX_DataExtractor/`** folder to any Windows PC or USB drive (no Python or PyTorch installation required on that PC) and launch the **Standalone Static Dataset Extractor**:

- **Launch Desktop GUI (Double-Click or Run in Terminal):**
  ```powershell
  .\dist\ASignX_DataExtractor\ASignX_DataExtractor.exe
  ```
  *(Or via Python: `python -m ml.extractor_gui`)*

- **Or Run in Headless CLI Mode (`.exe` or Python):**
  ```powershell
  .\dist\ASignX_DataExtractor\ASignX_DataExtractor.exe --input path/to/dataset_folder_or_zip --output ml/data/landmarks.csv
  # Or via Python module:
  python -m ml.extract_dataset --input path/to/dataset_folder_or_zip --output ml/data/landmarks.csv
  ```

**What the Extractor Automatically Does:**
- Accepts **any folder structure or `.zip` archive** (flat `A/img.jpg`, nested `train/a/img.jpg`, `test/Sign_B/img.png`),
- Canonicalizes labels case-insensitively (`a` / `sign_a` $\rightarrow$ `A`),
- Optionally consolidates accepted images into a clean `<staging>/<LABEL>/` folder (`--staging-dir`),
- Runs a **4-Pass Outlier & Duplicate Cleaner** (SHA-256 exact deduplication, anatomical bone-ratio plausibility check, statistical MAD outlier rejection, and $\epsilon$-ball near-duplicate pruning),
- Preserves source/session grouping; extracting or merging images does not guarantee three independent sessions per sign,
- **Non-destructively appends** clean samples to `ml/data/landmarks.csv` without overwriting existing data.

---

### Option B: Collect New Data Using Your Webcam

For guided recording without repeated commands, double-click **`ASignX_ManualTraining.bat`**, or launch:

```powershell
python -m ml.manual_training_gui
```

The Static Sign Studio includes camera preview, a three-second preparation countdown,
pause/resume, session selection, frame review and retakes. Choose a sign label, either
Left or Right hand, and a sample count (default 200 per session). Record **three sessions
total per sign**, not three per hand. Only the selected hand should be visible. Capture
pauses sample collection when the hand is missing or moving; the detector cannot verify
that you are making the correct sign, so review your frames. Change position, distance
or background between sessions. Captures from different days are more independent than
three consecutive captures in identical conditions.

After all three sessions are complete, choose **Add samples** or **Replace this sign**
and click **Process and save**. Processing re-extracts landmarks from original PNG frames,
requires at least 80% unique valid samples per session (minimum 10), rejects exact pose
conflicts and preserves handedness for canonical training. If a session fails, retake it
and retry. Replacement affects only the selected label and requires confirmation.

Working frames live in `ml/data/manual_recordings/pending/`, outside the training CSV.
Canceling or closing discards the whole unfinished batch; a forced termination leaves
cleanup for the next launch. Existing dataset entries and backup archives are preserved.
During processing, normal exit waits for the dataset transaction to finish. After a
verified ZIP backup and atomic CSV update, working recordings are deleted. Archives in
`ml/data/manual_recordings/backups/` contain original frames, session/camera settings,
extraction counts and the processed CSV. The previous target CSV is also saved beside
it as `.backup-<batch-id>`. Processing failures retain working frames for retry while
the app stays open. A backup made before an interrupted publication may contain an
uncommitted batch; check the CSV session IDs before restoring or importing it.

The dataset overview shows samples and sessions per label. **Train new version** checks
the existing three-session split rules and trains against an isolated CSV snapshot.
It saves the model, metrics, split manifest, snapshot and logs under a unique directory
in `ml/models/manual_runs/`; the deployed model is not overwritten. Training needs at
least two classes. Stop training preserves previous models; partial artifacts remain
in that run's folder. **Test model with camera** opens live recognition for the newly
trained or selected checkpoint (press Q to end the test). Use the new checkpoint path
explicitly when deploying and re-export ONNX for browser use.

The original command-line collector is still available:

```powershell
python -m ml.collect_data --label YOUR_LABEL --samples 200
```
- Press **Space** to start/pause recording.
- Press **Q** or **Esc** to finish early.
- Repeat for **at least 3 separate recording sessions** per sign.

---

### Option C: Convert a Standard Image Folder (`ml.prepare_images`)
```powershell
python -m ml.prepare_images --input images --output ml/data/landmarks.csv
```

---

### Step 2: Train the Static Model (`SignResMLP`)

New static training runs canonicalize handedness by default (`--static-hand-preprocessing left_x_reflection_v1`). Left-hand wrist-relative x coordinates are reflected into a shared Right representation before augmentation and kinematic enrichment. This preserves y/z and orientation cues. The original CSV, extractor, and dynamic/two-hand pipeline are unchanged.

The checkpoint records this input contract in `model_config.static_hand_preprocessing`. Live recognition, evaluation, registration, and coaching centroids use the same convention. Existing checkpoints without that field use legacy normalization; **retraining is required to activate canonicalization**. Use a separate output such as `ml/models/sign_classifier_canonical.pth` to compare with the existing model, then load it with `python -m ml.predict --sign-model ml/models/sign_classifier_canonical.pth`. Use `--static-hand-preprocessing none` only for a legacy comparison run.

Canonical training requires `Left` or `Right` handedness on every CSV row. Do not guess missing labels or reflect the CSV in place. Extraction and prediction must use detector labels and coordinates from the same image convention. Mirror a browser preview with CSS if desired; do not separately mirror model input or swap labels. Test both real hands after retraining; canonicalization reduces representation differences but does not guarantee robustness to occlusion or unseen poses.

For browser deployment, export the newly trained checkpoint again. Prepare both graph inputs and coaching inputs with `prepareStaticHandLandmarks(landmarks, handedness, metadata)` from `web_client/asignx_client_sdk.ts`. The exported `static_hand_preprocessing` field controls reflection; missing metadata means legacy inputs. The ONNX graph expects these externally prepared 63D features. Pass original detector landmarks into the helper exactly once.

Training requires **at least three independent recording/source sessions for every class**, with one or more sessions in each of train, validation, and test. A merged dataset invalidates the old split manifest; training regenerates it only when the updated dataset meets these requirements.

If training reports `need at least 3 independent sessions`, collect additional separate recordings for every affected label, or prepare a separate training CSV excluding unsupported classes. Do not rename row chunks into sessions: that can leak related samples across splits. Only correct source IDs when original provenance establishes independent sessions. For example, run `python -m ml.collect_data --label DEL --samples 200` in two separate recording sessions if DEL currently has one session.

Run the upgraded training pipeline (uses `AdamW`, `CosineAnnealingLR`, `label_smoothing=0.05`, 3D geometric landmark augmentation, and exports per-class 63D centroids for ASignLearn coaching):

```powershell
python -m ml.train --data ml/data/landmarks.csv --output ml/models/sign_classifier_canonical.pth --epochs 50 --batch-size 64
```

**What this produces:**
- `ml/models/sign_classifier_canonical.pth`: Trained `SignResMLP` checkpoint with 91D kinematic enricher, cosine prototype head, and `class_centroids`.
- `ml/models/sign_classifier_canonical.splits.json`: Leakage-free Train (70%), Validation (15%), and Test (15%) split manifest.
- `ml/models/sign_classifier_canonical.metrics.json`: Accuracy, precision, recall, per-class F1, and confusion matrix.

---

### Step 3: Evaluate Test Accuracy
```powershell
python -m ml.evaluate --data ml/data/landmarks.csv --model ml/models/sign_classifier_canonical.pth --split test
```

---

### Step 4: Test Live with Your Webcam (With On-Screen Sign Recognition HUD)
```powershell
# Static model only:
python -m ml.predict

# Unified Static + Dynamic recognition with MotionGate:
python main.py --sign-model ml/models/sign_classifier_canonical.pth --sequence-model ml/models/sequence_classifier.pth
```

---

## Part 2: Training Moving Signs (J, Z, Phrases)

### Standalone Dynamic Sign Studio (Windows GUI)

Launch `dist/ASignX_DynamicStudio.exe`, or double-click `ASignX_DynamicStudio.bat`.
Use **Webcam recording** to choose a label, preview the camera, and start a take
after a countdown. Click **Save take** after the complete gesture or **Discard take**
to retry. Use **Extract from videos** for labeled video folders. The single EXE
requires no Python installation on the recording PC and outputs `.npz` landmark
sequences rather than raw camera footage. Choose the output folder explicitly;
the portable app defaults to a `data` folder beside the EXE.

See [the Dynamic Sign Studio guide](docs/dynamic-studio-guide.md) for recording,
importing, and rebuilding. Source launch: `python -m ml.dynamic_studio_gui`.
The existing manual training launcher remains the static-sign studio.

### Option A: Capture Moving Gestures via Live Webcam
```powershell
python -m ml.collect_sequence --label J --takes 5
python -m ml.collect_sequence --label Z --takes 5
```

### Option B: Convert Existing Video Clips to Sequences
```powershell
python -m ml.prepare_videos --input videos --output ml/data/video_sequences
```

---

### Step 2: Train the Sequence Model (`TCN-Attention` or `LSTM`)
By default, `ml.train_sequence` trains the upgraded **Dilated TCN + Multi-Head Self-Attention (`tcn_attention`)** architecture with 3D sequence augmentation:

```powershell
# Train TCN-Attention (default backbone) on webcam sequences:
python -m ml.train_sequence --data-dir ml/data/sequences --output ml/models/sequence_classifier.pth --epochs 50

# Or on converted video sequences:
python -m ml.train_sequence --data-dir ml/data/video_sequences --output ml/models/sequence_classifier.pth --epochs 50
```

---

### Step 3: Test Moving Signs Live
```powershell
python -m ml.predict_sequence --model ml/models/sequence_classifier.pth
```

---

## Part 3: Client-Side Web Export for ASignLearn & ASignLive

To deploy your trained models directly inside the browser (using `@mediapipe/tasks-vision` + `onnxruntime-web` and `web_client/asignx_client_sdk.ts`):

```powershell
# Export Static SignResMLP bundle (.onnx / .traced.pt + .meta.json + .weights.json):
python -m ml.export_onnx --checkpoint ml/models/sign_classifier_canonical.pth --output web_client/models/sign_classifier.onnx --type static

# Export Dynamic Sequence bundle:
python -m ml.export_onnx --checkpoint ml/models/sequence_classifier.pth --output web_client/models/sequence_classifier.onnx --type sequence
```

---

## Part 4: Resetting / Starting from Scratch

- **Delete only the trained models (keep your dataset):**
  ```powershell
  Remove-Item -Force ml/models/sign_classifier.*
  ```

- **Delete all models and recorded data completely:**
  ```powershell
  Remove-Item -Force ml/models/*
  Remove-Item -Force ml/data/landmarks.csv
  Remove-Item -Recurse -Force ml/data/sequences -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force ml/data/video_sequences -ErrorAction SilentlyContinue
  ```

---

## Quick Reference Summary Table

| Task | Input | Output Format | Command |
|---|---|---|---|
| **Launch Portable Extractor GUI** | Folder or `.zip` | Appends clean rows to `landmarks.csv` | `.\dist\ASignX_DataExtractor\ASignX_DataExtractor.exe` |
| **Extract & clean via CLI** | Folder or `.zip` | Appends clean rows to `landmarks.csv` | `python -m ml.extract_dataset --input dataset.zip --output ml/data/landmarks.csv` |
| **Collect static poses** | Webcam | Rows in `ml/data/landmarks.csv` | `python -m ml.collect_data --label A --samples 200` |
| **Train static model (`SignResMLP`)** | `landmarks.csv` | `ml/models/sign_classifier_canonical.pth` | `python -m ml.train --epochs 50` |
| **Evaluate static model** | `landmarks.csv` | Metrics output | `python -m ml.evaluate --split test` |
| **Run live unified inference** | Webcam | Real-time console & HUD | `python -m ml.predict` |
| **Collect moving gestures** | Webcam | `.npz` files in `ml/data/sequences/` | `python -m ml.collect_sequence --label J --takes 5` |
| **Convert video dataset** | Folder of `.mp4` / `.mov` | `.npz` files in `ml/data/video_sequences/` | `python -m ml.prepare_videos --input videos --output ml/data/video_sequences` |
| **Train sequence model (`TCN-Attention`)** | `.npz` directory | `ml/models/sequence_classifier.pth` | `python -m ml.train_sequence --data-dir ml/data/sequences` |
| **Export for Web (`ASignLearn` / `ASignLive`)** | `.pth` checkpoint | `.onnx`, `.meta.json`, `.weights.json` | `python -m ml.export_onnx --checkpoint ml/models/sign_classifier_canonical.pth --output web_client/models/sign_classifier.onnx` |
