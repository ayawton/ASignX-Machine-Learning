"""Standalone GUI for recording and extracting moving-sign landmark sequences."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback

import cv2
from PIL import Image, ImageTk

from asignx import HandDetector, WebcamStream
from ml.dynamic_recording import CaptureSettings, TakeRecorder, validate_output
from ml.prepare_videos import prepare_video_dataset
from ml.sequence_io import load_sequence_file
from ml.video import HAND_SELECTIONS


def default_output(name: str) -> Path:
    # A portable app must never write into PyInstaller's temporary bundle directory.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data" / name
    return Path(__file__).resolve().parent.parent / "ml" / "data" / name


class DynamicStudio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events = queue.Queue()
        self.commands = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.worker = None
        self.mode = ""
        self.closing = False
        self.photo = None
        self.capture_state = "Ready"
        self.config_widgets = []
        self.label = tk.StringVar(value="J")
        self.hand = tk.StringVar(value="both")
        self.length = tk.StringVar(value="30")
        self.fps = tk.StringVar(value="15")
        self.minimum = tk.StringVar(value="10")
        self.ratio = tk.StringVar(value="0.5")
        self.maximum = tk.StringVar(value="10")
        self.countdown = tk.StringVar(value="3")
        self.camera = tk.StringVar(value="0")
        self.signer_id = tk.StringVar()
        self.recording_id = tk.StringVar()
        self.source_id = tk.StringVar()
        self.device_id = tk.StringVar()
        self.benchmark_split = tk.StringVar()
        self.output = tk.StringVar(value=str(default_output("sequences")))
        self.input = tk.StringVar()
        self.import_output = tk.StringVar(value=str(default_output("video_sequences")))
        self.skip_invalid = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready. Choose a label and open the camera, or import labeled videos.")
        self.quality = tk.StringVar(value="Camera closed")
        root.title("ASignX - Dynamic Sign Studio")
        root.geometry("1080x850")
        root.minsize(980, 760)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build()
        root.after(50, self.poll)

    def field(self, parent, row, title, variable, choices=None):
        ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        widget = (ttk.Combobox(parent, textvariable=variable, values=choices, state="readonly", width=18)
                  if choices else ttk.Entry(parent, textvariable=variable, width=21))
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        self.config_widgets.append((widget, "readonly" if choices else "normal"))

    def folder(self, parent, variable, title):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text=title, width=16).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True, padx=5)
        button = ttk.Button(row, text="Browse...", command=lambda: self.browse(variable))
        button.pack(side="left")
        self.config_widgets.extend(((entry, "normal"), (button, "normal")))

    def browse(self, variable):
        selected = filedialog.askdirectory(parent=self.root, title="Choose a folder", mustexist=False)
        if selected:
            variable.set(selected)

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", padding=(10, 7))
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Dynamic Sign Studio", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Record complete moving signs or extract labeled videos into training-ready landmark sequences.").pack(anchor="w", pady=(3, 10))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        record = ttk.Frame(self.tabs, padding=12)
        imported = ttk.Frame(self.tabs, padding=12)
        identity = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(record, text="  Webcam recording  ")
        self.tabs.add(imported, text="  Extract from videos  ")
        self.tabs.add(identity, text="  Recording identity  ")
        ttk.Label(identity, text="Identify the person consistently across every sign and take.\nLeave unknown fields blank; separate take IDs do not establish independent people.\nThese settings apply to new webcam recordings. Video imports need a separate reviewed identity plan.", justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 18))
        for row, (title, variable, choices) in enumerate([
            ("Person ID", self.signer_id, None), ("Recording visit ID", self.recording_id, None),
            ("Source corpus ID", self.source_id, None), ("Device ID", self.device_id, None),
            ("Reserved split", self.benchmark_split, ("", "train", "validation", "test")),
        ], start=1):
            self.field(identity, row, title, variable, choices)
        self.folder(record, self.output, "Save sequences to")
        body = ttk.Frame(record)
        body.pack(fill="both", expand=True, pady=8)
        settings = ttk.LabelFrame(body, text="Recording settings", padding=10)
        settings.pack(side="left", fill="y", padx=(0, 12))
        for row, (title, variable, choices) in enumerate([
            ("Sign label", self.label, None), ("Hand mode", self.hand, HAND_SELECTIONS),
            ("Camera index", self.camera, None), ("Sequence steps", self.length, None),
            ("Sampling FPS", self.fps, None), ("Minimum frames", self.minimum, None),
            ("Min. detection ratio", self.ratio, None), ("Max. take seconds", self.maximum, None),
            ("Countdown seconds", self.countdown, None),
        ]):
            self.field(settings, row, title, variable, choices)
        ttk.Label(settings, text="Keep settings consistent across labels.\nRecord 3+ takes per label and 2+ labels.\nVary signer, lighting and position.", wraplength=280).grid(row=10, column=0, columnspan=2, sticky="w", pady=8)
        preview = ttk.Frame(body)
        preview.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(preview, text="Open camera to preview", bg="#101827", fg="#cbd5e1", font=("Segoe UI", 15))
        self.preview.pack(fill="both", expand=True)
        ttk.Label(preview, textvariable=self.quality, font=("Segoe UI", 11, "bold")).pack(pady=8)
        controls = ttk.Frame(record)
        controls.pack(fill="x")
        self.open_button = ttk.Button(controls, text="Open camera", command=self.open_camera)
        self.take_button = ttk.Button(controls, text="Start take", command=self.toggle_take, state="disabled")
        self.discard_button = ttk.Button(controls, text="Discard take", command=lambda: self.commands.put("discard"), state="disabled")
        self.stop_button = ttk.Button(controls, text="Close camera", command=self.stop_work, state="disabled")
        for button in (self.open_button, self.take_button, self.discard_button, self.stop_button):
            button.pack(side="left", padx=(0, 8))
        ttk.Label(record, text="Start take -> countdown -> perform the entire sign -> Save take. Maximum duration saves automatically.").pack(anchor="w", pady=(10, 2))
        ttk.Label(record, text="Saves extracted .npz landmarks only. Raw camera video is not retained; saved takes are kept when you close.").pack(anchor="w")
        ttk.Label(imported, text="1. Record one complete sign per video.\n2. Put videos in folders such as videos/J/take1.mp4 and videos/Z/take1.mp4.\n3. Choose the videos folder below. At least 2 labels and 3 unique videos per label are required.", justify="left", font=("Segoe UI", 11)).pack(anchor="w", pady=(6, 20))
        self.folder(imported, self.input, "Labeled video root")
        self.folder(imported, self.import_output, "Save sequences to")
        ttk.Label(imported, text="Import uses the Hand mode, Sequence steps, Sampling FPS, and Min. detection ratio from the recording tab.\nUse a separate output folder for imported videos. Existing matching files are reused; incompatible files cause an error.", wraplength=900).pack(anchor="w", pady=15)
        check = ttk.Checkbutton(imported, text="Skip invalid videos (still requires 3 valid unique videos per label)", variable=self.skip_invalid)
        check.pack(anchor="w")
        self.config_widgets.append((check, "normal"))
        actions = ttk.Frame(imported)
        actions.pack(fill="x", pady=18)
        self.import_button = ttk.Button(actions, text="Extract videos", command=self.import_videos)
        self.import_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Stop after current video", command=self.stop_work, state="disabled")
        self.cancel_button.pack(side="left", padx=10)
        self.progress = ttk.Progressbar(imported, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=10)
        ttk.Label(imported, text="MP4, AVI, MOV, MKV, M4V, and WEBM are supported. Training remains a separate step in the project.").pack(anchor="w")
        ttk.Separator(outer).pack(fill="x", pady=10)
        ttk.Label(outer, textvariable=self.status, wraplength=1000).pack(anchor="w")
        self.log = tk.Text(outer, height=4, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="x", pady=(8, 0))

    def settings(self) -> CaptureSettings:
        if not self.output.get().strip():
            raise ValueError("Choose an output folder.")
        return CaptureSettings(Path(self.output.get()).expanduser().resolve(), self.label.get(), self.hand.get(),
                               int(self.length.get()), float(self.fps.get()), int(self.minimum.get()),
                               float(self.ratio.get()), float(self.maximum.get()), float(self.countdown.get()),
                               signer_id=self.signer_id.get().strip(), recording_id=self.recording_id.get().strip(),
                               source_id=self.source_id.get().strip(), device=self.device_id.get().strip(),
                               split=self.benchmark_split.get())

    def append(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", str(message) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, mode):
        self.mode = mode
        for widget, idle_state in self.config_widgets:
            widget.configure(state="disabled" if mode else idle_state)
        self.open_button.configure(state="disabled" if mode else "normal")
        self.import_button.configure(state="disabled" if mode else "normal")
        self.stop_button.configure(state="normal" if mode == "camera" else "disabled")
        self.cancel_button.configure(state="normal" if mode == "import" else "disabled")
        if mode != "camera":
            self.take_button.configure(state="disabled")
            self.discard_button.configure(state="disabled")

    def start_worker(self, mode, target):
        self.stop.clear()
        self.commands = queue.Queue()
        self.set_busy(mode)
        def run():
            try:
                target()
            except InterruptedError as error:
                self.events.put(("message", str(error)))
            except Exception as error:
                self.events.put(("error", f"{error}\n\n{traceback.format_exc()}"))
            finally:
                self.events.put(("done", None))
        self.worker = threading.Thread(target=run, name="ASignX-DynamicStudio", daemon=False)
        self.worker.start()

    def open_camera(self):
        if self.mode:
            return
        try:
            settings = self.settings()
            settings.validate()
            camera_index = int(self.camera.get())
            if camera_index < 0:
                raise ValueError("Camera index must be nonnegative.")
        except (ValueError, OSError) as error:
            messagebox.showerror("Check recording settings", str(error), parent=self.root)
            return
        self.status.set("Opening camera...")
        self.start_worker("camera", lambda: self.camera_worker(settings, camera_index))

    def camera_worker(self, settings, camera_index):
        recorder = TakeRecorder(settings)
        with WebcamStream(camera_index=camera_index, threaded=False, exposure=None) as camera:
            with HandDetector(max_num_hands=2) as detector:
                self.events.put(("camera_ready", None))
                while not self.stop.is_set():
                    success, frame = camera.read()
                    if self.stop.is_set():
                        break
                    if not success or frame is None:
                        raise RuntimeError("Camera stopped delivering frames. Reconnect it and open the camera again.")
                    result, annotated = detector.process(frame, draw=True)
                    now = time.monotonic()
                    try:
                        while not self.commands.empty():
                            command = self.commands.get_nowait()
                            if command == "start":
                                recorder.start(now)
                            elif command == "save":
                                self.events.put(("saved", str(recorder.save(now))))
                            elif command == "discard":
                                recorder.discard()
                                self.events.put(("message", "Take discarded. Press Start take to retry."))
                        saved = recorder.update(result.all_hands, now)
                        if saved:
                            self.events.put(("saved", str(saved)))
                    except ValueError as error:
                        self.events.put(("message", f"Take not saved: {error}"))
                    self.events.put(("capture", (recorder.state, recorder.describe(now))))
                    # A bounded queue avoids lag and memory growth if the UI is busy.
                    if self.frames.empty():
                        image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        self.frames.put_nowait(image)
                if recorder.state != "Ready":
                    recorder.discard()
                    self.events.put(("message", "Unfinished take discarded. Previously saved takes are kept."))

    def toggle_take(self):
        self.commands.put("save" if self.capture_state == "Recording" else "start")
        self.take_button.configure(state="disabled")

    def import_videos(self):
        if self.mode:
            return
        try:
            if not self.input.get().strip() or not self.import_output.get().strip():
                raise ValueError("Choose both the labeled video root and output folder.")
            source = Path(self.input.get()).expanduser().resolve()
            output = Path(self.import_output.get()).expanduser().resolve()
            if not source.is_dir():
                raise ValueError("The labeled video root must be an existing folder.")
            if source == output or source in output.parents or output in source.parents:
                raise ValueError("Choose separate, non-nested input and output folders.")
            hand, length, fps, ratio = self.hand.get(), int(self.length.get()), float(self.fps.get()), float(self.ratio.get())
            if length < 2 or not 0 < fps <= 120 or not 0 < ratio <= 1:
                raise ValueError("Check sequence steps, sampling FPS, and detection ratio.")
            validate_output(output, hand, length)
            skip = self.skip_invalid.get()
        except (ValueError, OSError) as error:
            messagebox.showerror("Check import settings", str(error), parent=self.root)
            return
        self.progress["value"] = 0
        self.status.set("Preparing video extraction...")
        def work():
            paths = prepare_video_dataset(source, output, hand_selection=hand, sequence_length=length,
                target_fps=fps, min_detection_ratio=ratio, skip_invalid=skip,
                progress=lambda done, total, path: self.events.put(("progress", (done, total, path))),
                cancelled=self.stop.is_set)
            self.events.put(("message", f"Extraction complete: {len(paths)} sequences in {output}"))
        self.start_worker("import", work)

    def stop_work(self):
        self.stop.set()
        self.take_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.status.set("Closing camera..." if self.mode == "camera" else "Stopping after the current video; completed sequences will be kept...")

    def poll(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind in ("message", "saved"):
                message = f"Saved: {value}" if kind == "saved" else value
                self.status.set(message)
                self.append(message)
            elif kind == "camera_ready":
                self.status.set("Camera ready. Press Start take for the preparation countdown.")
                self.take_button.configure(state="normal")
                self.discard_button.configure(state="normal")
            elif kind == "capture":
                self.capture_state, description = value
                self.quality.set(description)
                self.take_button.configure(text="Save take" if self.capture_state == "Recording" else "Start take",
                    state="disabled" if self.capture_state == "Preparing" or self.stop.is_set() else "normal")
            elif kind == "progress":
                done, total, path = value
                self.progress["value"] = 100 * done / max(total, 1)
                self.status.set(f"{done}/{total}: {path}")
            elif kind == "error":
                self.status.set(str(value).split("\n")[0])
                self.append(value)
                if not self.closing:
                    messagebox.showerror("Operation could not finish", str(value).split("\n\n")[0], parent=self.root)
            elif kind == "done":
                self.set_busy("")
                self.quality.set("Camera closed")
                self.preview.configure(image="", text="Open camera to preview")
                self.photo = None
                while not self.frames.empty():
                    self.frames.get_nowait()
        if not self.frames.empty() and self.mode == "camera":
            image = Image.fromarray(self.frames.get_nowait())
            image.thumbnail((max(320, self.preview.winfo_width()), max(240, self.preview.winfo_height())))
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="")
        if self.closing and not self.mode:
            self.root.destroy()
            return
        self.root.after(50, self.poll)

    def close(self):
        if self.mode:
            self.closing = True
            self.stop_work()
        else:
            self.root.destroy()


def self_test(output: Path):
    """Exercise bundled Tk, OpenCV, MediaPipe and storage without a camera."""
    import numpy as np
    root = tk.Tk()
    root.withdraw()
    app = DynamicStudio(root)
    root.update()
    ImageTk.PhotoImage(Image.new("RGB", (16, 16)))
    with HandDetector(max_num_hands=2) as detector:
        result, _ = detector.process(np.zeros((64, 64, 3), dtype=np.uint8), draw=True)
    from types import SimpleNamespace
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        recorder = TakeRecorder(CaptureSettings(Path(directory), "SMOKE", min_frames=2, countdown_seconds=0))
        recorder.start(0)
        for step in range(3):
            points = np.zeros((21, 3), dtype=np.float32)
            points[:, 0] = np.linspace(0, 0.2, 21) + step * 0.03
            points[:, 1] = np.linspace(0, 0.1, 21)
            recorder.update([SimpleNamespace(handedness="Right", to_numpy=lambda p=points: p)], step * 0.1)
        sample = load_sequence_file(recorder.save(0.3))
        assert sample.features.shape == (30, 126)
    root.destroy()
    output.write_text(json.dumps({"ok": True, "tk": tk.TkVersion, "opencv": cv2.__version__,
                                 "blank_frame_hands": len(result.all_hands), "shape": [30, 126]}), encoding="utf-8")


def main():
    # Windowed executables have no stdout/stderr; existing extraction code prints progress.
    for stream in ("stdout", "stderr"):
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w", encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        try:
            self_test(args.self_test)
        except Exception:
            args.self_test.write_text(json.dumps({"ok": False, "error": traceback.format_exc()}), encoding="utf-8")
            raise
        return
    root = tk.Tk()
    DynamicStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
