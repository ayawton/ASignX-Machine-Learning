# ASignX - Hand Recognition Module

The **Hand Recognition Module** is the foundational computer-vision component of the **ASignX** Sign Language Recognition System. Its sole responsibility is to detect and track a user's hand (supporting single-hand and dual-hand tracking) through a webcam and extract its 21 3D hand landmarks in real time at fluid **>=30 FPS**.

> **Scope Note:** This module strictly handles detection, tracking, and coordinate extraction. Sign classification (e.g. `HELLO`, `YES`, `NO`) is intentionally decoupled and performed by downstream modules.

---

## Pipeline

```text
Webcam (Threaded Asynchronous Frame Grabber)
   ↓
Video Frame (OpenCV BGR, Buffer Size 1, Exposure-Optimized)
   ↓
MediaPipe Hands (static_image_mode=False, max_num_hands=2)
   ↓
Hand Detection (Confidence check for up to 2 hands)
   ↓
21 Hand Landmarks per hand (Preserved indices 0..20)
   ↓
Structured Landmark Coordinates (Left & Right hands)
```

---

## 21 Hand Landmarks Map

| Index | Landmark Name | Region |
|:-----:|:--------------|:-------|
| 0 | `WRIST` | Wrist |
| 1 | `THUMB_CMC` | Thumb carpometacarpal |
| 2 | `THUMB_MCP` | Thumb metacarpophalangeal |
| 3 | `THUMB_IP` | Thumb interphalangeal |
| 4 | `THUMB_TIP` | Thumb tip |
| 5 | `INDEX_FINGER_MCP` | Index metacarpophalangeal |
| 6 | `INDEX_FINGER_PIP` | Index proximal interphalangeal |
| 7 | `INDEX_FINGER_DIP` | Index distal interphalangeal |
| 8 | `INDEX_FINGER_TIP` | Index tip |
| 9 | `MIDDLE_FINGER_MCP` | Middle metacarpophalangeal |
| 10 | `MIDDLE_FINGER_PIP` | Middle proximal interphalangeal |
| 11 | `MIDDLE_FINGER_DIP` | Middle distal interphalangeal |
| 12 | `MIDDLE_FINGER_TIP` | Middle tip |
| 13 | `RING_FINGER_MCP` | Ring metacarpophalangeal |
| 14 | `RING_FINGER_PIP` | Ring proximal interphalangeal |
| 15 | `RING_FINGER_DIP` | Ring distal interphalangeal |
| 16 | `RING_FINGER_TIP` | Ring tip |
| 17 | `PINKY_MCP` | Pinky metacarpophalangeal |
| 18 | `PINKY_PIP` | Pinky proximal interphalangeal |
| 19 | `PINKY_DIP` | Pinky distal interphalangeal |
| 20 | `PINKY_TIP` | Pinky tip |

---

## Structured Output

### Dual-Hand Output (When 2 hands are detected):

```text
Hands (2 detected)
├── Hand (Left)
│   ├── Landmark  0 (WRIST            ) → (+0.4512, +0.6210, -0.0120)
│   ├── Landmark  1 (THUMB_CMC        ) → (+0.4105, +0.5821, -0.0185)
│   ├── ...
│   └── Landmark 20 (PINKY_TIP        ) → (+0.3182, +0.3411, -0.0380)
└── Hand (Right)
    ├── Landmark  0 (WRIST            ) → (+0.7523, +0.6514, -0.0152)
    ├── Landmark  1 (THUMB_CMC        ) → (+0.7931, +0.5902, -0.0211)
    ├── ...
    └── Landmark 20 (PINKY_TIP        ) → (+0.8845, +0.3620, -0.0410)
```

### Single-Hand Output (When 1 hand is detected):

```text
Hand (Right)
├── Landmark  0 (WRIST            ) → (+0.5213, +0.7429, -0.0124)
├── Landmark  1 (THUMB_CMC        ) → (+0.4612, +0.6811, -0.0210)
├── ...
└── Landmark 20 (PINKY_TIP        ) → (+0.3245, +0.3418, -0.0452)
```

### No-Hand Output:

```text
Hand → null
```
*(No artificial landmark values are generated).*

---

## Performance Optimizations (Reaching 30+ FPS)

1. **Threaded Frame Acquisition (`WebcamStream(threaded=True)`)**:
   Decouples OpenCV hardware frame capture into a dedicated background daemon thread. The inference loop immediately reads the newest frame in `<0.05ms` without blocking for sensor exposure.
2. **Camera Exposure Lock (`--exposure -5.0`)**:
   Standard Windows laptop webcams automatically drop framerate to 15 FPS (or 7.5 FPS at 720p) in indoor lighting due to auto-exposure priority. Setting DirectShow exposure to `-5.0` ($2^{-5} = 1/32\text{s}$) locks in **30 FPS**.
3. **Optimized Resolution & Complexity Defaults**:
   - Default resolution: `640x480` for smooth >=30-60 FPS performance.
   - Model complexity: `1` (full accuracy) or `0` (ultra-fast lite model).
4. **Motion-Aware Left-Hand Reacquisition**:
   - The most recent left-hand motion is extrapolated across the next frame.
   - A bounded 0.45 normalized-frame association gate preserves the `Left`
     label after a rapid sweep, without emitting stale landmarks during an
     actual detection loss.

---

## Quickstart

### 1. Installation

Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Real-Time Visual Debugging App

```bash
python main.py
```

Options:
```bash
python main.py --help
  --camera CAMERA           Webcam device index (default: 0)
  --width WIDTH             Webcam width (default: 640 for >=30 FPS)
  --height HEIGHT           Webcam height (default: 480 for >=30 FPS)
  --max-hands MAX_HANDS     Maximum hands to track (default: 2)
  --model-complexity {0,1}  0=Lite (fastest), 1=Full (default: 1)
  --min-detection-confidence FLOAT  Detection threshold (default: 0.6 for rapid reacquisition)
  --exposure EXPOSURE       Exposure: '-5.0' (locks 30 FPS), '-6.0' (60 FPS), or 'auto' (default: -5.0)
  --no-threads              Disable background threaded reader
  --print-stream            Continuously stream landmark tree to terminal
```

#### Keyboard Controls (Visual Window)

* **`q`** or **`Esc`**: Exit application cleanly.
* **`d`**: Toggle hand landmark points and skeleton overlay on/off.
* **`h`**: Toggle HUD status banner and key coordinate panels on/off.
* **`e`**: Cycle exposure mode between **30 FPS locked (-5.0)**, **60 FPS locked (-6.0)**, and **Auto-Exposure**.
* **`p`**: Print the current frame's complete structured landmark tree to terminal.
* **`s`**: Save current visual snapshot (`.jpg`) and structured landmark coordinates (`.json`) into `snapshots/`.

---

## Programmatic API Usage

```python
from asignx import WebcamStream, HandDetector

# 1. Initialize threaded camera (locked at 30 FPS) and 2-hand detector
with WebcamStream(camera_index=0, width=640, height=480, threaded=True, exposure=-5.0) as camera:
    with HandDetector(max_num_hands=2) as detector:
        for frame in camera.stream():
            # 2. Process frame
            result, annotated_frame = detector.process(frame, draw=True)

            # 3. Access hands
            if result.hand_detected:
                print(f"Detected {result.hands_count} hand(s)")
                
                # Direct access to specific hands
                if result.left_hand is not None:
                    print(f"Left hand wrist: {result.left_hand[0].as_tuple()}")
                if result.right_hand is not None:
                    print(f"Right hand wrist: {result.right_hand[0].as_tuple()}")

                # Export to NumPy array (shape: 21, 3 per hand)
                coords_np = result.hand.to_numpy()

                # Export to JSON
                json_output = result.to_json()
            else:
                print("Hand -> null")
```

---

## Running Tests

Execute the automated test suite with pytest:

```bash
pytest -v
```
