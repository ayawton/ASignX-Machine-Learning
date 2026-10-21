"""Guided static-sign studio. Launch with python -m ml.manual_training_gui."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ml.manual_recording import (
    RecordingBatch, RecordingWorkspace, dataset_overview, identifier, process_batch,
)

PROJECT = Path(__file__).resolve().parent.parent


class ManualTrainingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.workspace = RecordingWorkspace(PROJECT / 'ml/data/manual_recordings')
        removed = self.workspace.cleanup_abandoned()
        self.batch = None
        self.camera = self.detector = None
        self.busy = False
        self.closing = False
        self.recording = False
        self.countdown = 0.0
        self.last_sample = 0.0
        self.motion = deque(maxlen=3)
        self.events = queue.Queue()
        self.training = None
        self.train_stop = threading.Event()
        self.test_process = None
        self.test_log = None
        self.test_log_path = None
        self.model = None
        self.photo = None
        self.csv = tk.StringVar(value=str(PROJECT / 'ml/data/landmarks.csv'))
        self.label = tk.StringVar()
        self.hand = tk.StringVar(value='Right')
        self.samples = tk.IntVar(value=200)
        self.camera_index = tk.IntVar(value=0)
        self.session = tk.StringVar(value='1')
        self.mode = tk.StringVar(value='Add samples')
        self.epochs = tk.IntVar(value=50)
        self.batch_size = tk.IntVar(value=64)
        self.status = tk.StringVar(value=f'Ready. Removed {removed} abandoned batch(es).' if removed else 'Ready. Choose a label and open the camera.')
        self.quality = tk.StringVar(value='Camera closed')
        self.progress = tk.StringVar(value='No active recording batch')
        self.training_progress = tk.StringVar(value='No training run active')
        root.title('ASignX — Guided Static Sign Training')
        root.geometry('1120x900')
        root.minsize(1000, 820)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.build()
        self.refresh()
        root.after(60, self.tick)
        root.after(100, self.poll)

    def build(self):
        outer = ttk.Frame(self.root, padding=15)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='Static Sign Studio', font=('Segoe UI', 21, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='Three sessions per sign, using either hand. Record only poses whose meaning is preserved when switching hands.').pack(anchor='w', pady=(0, 10))
        row = ttk.Frame(outer)
        row.pack(fill='x')
        ttk.Label(row, text='Dataset CSV').pack(side='left')
        self.csv_entry = ttk.Entry(row, textvariable=self.csv)
        self.csv_entry.pack(side='left', fill='x', expand=True, padx=8)
        self.browse_button = ttk.Button(row, text='Browse', command=self.browse)
        self.browse_button.pack(side='left')
        self.refresh_button = ttk.Button(row, text='Refresh overview', command=self.refresh)
        self.refresh_button.pack(side='left', padx=5)
        body = ttk.Frame(outer)
        body.pack(fill='both', expand=True, pady=10)
        left = ttk.Frame(body)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(body, width=310)
        right.pack(side='right', fill='y', padx=(15, 0))
        preview_frame = ttk.Frame(left, height=260, width=640)
        preview_frame.pack(fill='both', expand=True)
        preview_frame.pack_propagate(False)
        self.preview = ttk.Label(preview_frame, text='Open camera to preview', anchor='center', background='#172033')
        self.preview.pack(fill='both', expand=True)
        ttk.Label(left, textvariable=self.quality).pack(anchor='w', pady=5)
        settings = ttk.Frame(left)
        settings.pack(fill='x')
        self.settings = []
        for col, (name, variable, values) in enumerate([
            ('Sign label', self.label, None), ('Hand', self.hand, ('Right', 'Left')),
            ('Samples/session', self.samples, None), ('Camera', self.camera_index, None),
        ]):
            ttk.Label(settings, text=name).grid(row=0, column=col, sticky='w', padx=4)
            widget = ttk.Combobox(settings, textvariable=variable, values=values, state='readonly', width=12) if values else ttk.Entry(settings, textvariable=variable, width=15)
            widget.grid(row=1, column=col, padx=4, pady=4, sticky='ew')
            self.settings.append(widget)
        self.open_button = ttk.Button(left, text='Open / close camera', command=self.toggle_camera)
        self.open_button.pack(anchor='w', pady=5)
        self.new_button = ttk.Button(left, text='Start three-session batch', command=self.new_batch)
        self.new_button.pack(anchor='w')
        controls = ttk.Frame(left)
        controls.pack(fill='x', pady=8)
        ttk.Label(controls, text='Session').pack(side='left')
        self.session_box = ttk.Combobox(controls, textvariable=self.session, values=('1', '2', '3'), state='readonly', width=4)
        self.session_box.pack(side='left', padx=5)
        self.session_box.bind('<<ComboboxSelected>>', self.select_session)
        self.record_button = ttk.Button(controls, text='Record / pause', command=self.record)
        self.record_button.pack(side='left', padx=3)
        self.retake_button = ttk.Button(controls, text='Retake session', command=self.retake)
        self.retake_button.pack(side='left', padx=3)
        self.review_button = ttk.Button(controls, text='Review frames', command=self.review)
        self.review_button.pack(side='left', padx=3)
        self.bar = ttk.Progressbar(left, maximum=100)
        self.bar.pack(fill='x')
        ttk.Label(left, textvariable=self.progress, wraplength=680).pack(anchor='w', pady=5)
        ttk.Label(left, text='Between sessions: relax your hand, then vary angle, distance or background. Separate days provide stronger evaluation.', wraplength=680).pack(anchor='w')
        actions = ttk.Frame(left)
        actions.pack(fill='x', pady=8)
        self.mode_box = ttk.Combobox(actions, textvariable=self.mode, values=('Add samples', 'Replace this sign'), state='readonly', width=18)
        self.mode_box.pack(side='left')
        self.process_button = ttk.Button(actions, text='Process and save', command=self.process)
        self.process_button.pack(side='left', padx=5)
        self.discard_button = ttk.Button(actions, text='Discard batch', command=self.discard)
        self.discard_button.pack(side='left')
        ttk.Label(right, text='Dataset overview', font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        self.overview = ttk.Treeview(right, columns=('label', 'samples', 'sessions'), show='headings', height=11)
        for key, width in [('label', 100), ('samples', 80), ('sessions', 70)]:
            self.overview.heading(key, text=key.title())
            self.overview.column(key, width=width, anchor='center')
        self.overview.pack(fill='x', pady=8)
        ttk.Label(right, text='Training', font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        for title, variable in [('Epochs', self.epochs), ('Batch size', self.batch_size)]:
            row = ttk.Frame(right)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=title).pack(side='left')
            ttk.Entry(row, textvariable=variable, width=8).pack(side='right')
        self.train_button = ttk.Button(right, text='Train new version', command=self.train)
        self.train_button.pack(fill='x', pady=8)
        self.train_bar = ttk.Progressbar(right, maximum=100)
        self.train_bar.pack(fill='x', pady=3)
        ttk.Label(right, textvariable=self.training_progress, wraplength=290).pack(anchor='w')
        self.stop_button = ttk.Button(right, text='Stop training', command=self.stop_training)
        self.stop_button.pack(fill='x')
        self.test_button = ttk.Button(right, text='Test model with camera', command=self.test)
        self.test_button.pack(fill='x', pady=8)
        ttk.Button(right, text='Select model to test', command=self.choose_model).pack(fill='x')
        ttk.Label(right, text='Models, dataset snapshots and logs are saved under ml/models/manual_runs/. Recording archives are under ml/data/manual_recordings/backups/.', wraplength=290).pack(anchor='w', pady=8)
        self.log = tk.Text(outer, height=7, state='disabled', wrap='word', font=('Consolas', 9))
        self.log.pack(fill='x')
        ttk.Label(outer, textvariable=self.status, wraplength=1050).pack(anchor='w', pady=5)
        self.controls()

    def write(self, message):
        self.log.configure(state='normal')
        self.log.insert('end', str(message).rstrip() + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def controls(self):
        active = self.batch is not None
        for widget in [self.csv_entry, self.browse_button, *self.settings]:
            widget.configure(state='disabled' if active or self.busy else ('readonly' if isinstance(widget, ttk.Combobox) else 'normal'))
        for widget in [self.record_button, self.retake_button, self.review_button, self.discard_button]:
            widget.configure(state='normal' if active and not self.busy else 'disabled')
        self.session_box.configure(state='readonly' if active and not self.busy else 'disabled')
        self.new_button.configure(state='disabled' if active or self.busy else 'normal')
        self.process_button.configure(state='normal' if active and not self.busy and all(s['complete'] for s in self.batch.data['sessions']) else 'disabled')
        self.mode_box.configure(state='disabled' if self.busy else 'readonly')
        self.train_button.configure(state='disabled' if active or self.busy else 'normal')
        self.open_button.configure(state='disabled' if self.busy else 'normal')
        self.stop_button.configure(state='normal' if self.training else 'disabled')
        self.test_button.configure(state='normal' if self.model and not active and not self.busy else 'disabled')

    def browse(self):
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')], initialfile='landmarks.csv')
        if path:
            self.csv.set(path)
            self.refresh()

    def refresh(self):
        if self.busy:
            return
        try:
            rows = dataset_overview(self.csv.get())
            self.overview.delete(*self.overview.get_children())
            for row in rows:
                self.overview.insert('', 'end', values=row)
        except Exception as error:
            self.status.set(f'Dataset check: {error}')

    def toggle_camera(self):
        if self.test_process and self.test_process.poll() is None:
            self.status.set('Finish the model camera test with Q before opening the recording camera.')
            return
        if self.camera:
            self.close_camera()
            return
        try:
            from asignx import HandDetector, WebcamStream
            self.camera = WebcamStream(camera_index=self.camera_index.get(), exposure=None).start()
            self.detector = HandDetector(max_num_hands=2)
            self.status.set('Camera ready. Hold the selected sign, then record.')
        except Exception as error:
            self.close_camera()
            messagebox.showerror('Camera unavailable', str(error))

    def close_camera(self):
        self.recording = False
        self.countdown = 0
        if self.camera:
            self.camera.stop()
            self.camera = None
        if self.detector:
            self.detector.close()
            self.detector = None
        self.quality.set('Camera closed')

    def new_batch(self):
        try:
            self.batch = RecordingBatch(self.workspace, self.label.get(), self.hand.get(), self.samples.get(), self.camera_index.get())
            self.session.set('1')
            self.status.set('Session 1 ready. Record all three sessions before processing.')
            self.update_progress()
            self.controls()
        except Exception as error:
            messagebox.showerror('Cannot start batch', str(error))

    def select_session(self, *_):
        self.recording = False
        self.countdown = 0
        self.motion.clear()
        self.update_progress()

    def update_progress(self):
        if not self.batch:
            self.progress.set('No active recording batch')
            self.bar['value'] = 0
            return
        sessions = self.batch.data['sessions']
        selected = sessions[int(self.session.get()) - 1]
        total = self.batch.data['samples']
        self.bar['value'] = selected['frames'] / total * 100
        self.progress.set(' | '.join(f"Session {i}: {'complete' if s['complete'] else str(s['frames']) + '/' + str(total)}" for i, s in enumerate(sessions, 1)))

    def record(self):
        if not self.batch:
            return
        if self.batch.data['sessions'][int(self.session.get()) - 1]['complete']:
            self.status.set('This session is complete. Select another session or use Retake.')
            return
        if not self.camera:
            self.toggle_camera()
        if not self.camera:
            return
        if self.recording or self.countdown:
            self.recording = False
            self.countdown = 0
            self.status.set('Paused. Recorded samples are retained within this unfinished batch.')
        else:
            self.countdown = time.monotonic() + 3
            self.motion.clear()

    def retake(self):
        if self.batch and messagebox.askyesno('Retake session', 'Delete this session’s working frames and record it again?'):
            self.select_session()
            self.batch.retake(int(self.session.get()) - 1)
            self.update_progress()
            self.controls()

    def review(self):
        if not self.batch:
            return
        self.select_session()
        frames = sorted(self.batch.session_path(int(self.session.get()) - 1).glob('*.png'))
        if not frames:
            self.status.set('Record some samples before reviewing.')
            return
        window = tk.Toplevel(self.root)
        window.title('Review session — use the slider to inspect captured poses')
        preview = ttk.Label(window)
        preview.pack()
        def show(value):
            preview.image = tk.PhotoImage(file=str(frames[int(float(value))]))
            preview.configure(image=preview.image)
        tk.Scale(window, from_=0, to=len(frames) - 1, orient='horizontal', length=600, command=show).pack()
        show(0)

    def tick(self):
        if self.camera and not self.busy:
            try:
                import cv2
                import numpy as np
                from ml.preprocessing import normalize_landmarks
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    raise RuntimeError('Camera stopped delivering frames.')
                result, annotated = self.detector.process(frame.copy(), draw=True)
                hands = [h for h in result.all_hands if h.handedness == self.hand.get()]
                valid = len(result.all_hands) == 1 and len(hands) == 1
                quality = 'Show only the selected hand' if not valid else 'Hand visible — hold the sign steady'
                if valid:
                    points = hands[0].to_numpy()
                    normalize_landmarks(points)
                    self.motion.append(points[[0, 4, 8, 12, 16, 20]])
                    if len(self.motion) >= 2:
                        moving = np.linalg.norm(self.motion[-1] - self.motion[-2], axis=1).mean() > 0.025
                        if moving:
                            valid = False
                            quality = 'Movement detected — hold still to capture'
                else:
                    self.motion.clear()
                now = time.monotonic()
                if self.countdown:
                    if now >= self.countdown:
                        self.countdown = 0
                        self.recording = True
                    else:
                        quality = f'Get ready: {int(self.countdown - now) + 1}'
                if self.recording and self.batch and valid and now - self.last_sample >= 0.15:
                    index = int(self.session.get()) - 1
                    count = self.batch.add_frame(index, frame)
                    self.last_sample = now
                    self.update_progress()
                    if count >= self.batch.data['samples']:
                        self.batch.finish_session(index)
                        self.recording = False
                        self.update_progress()
                        self.controls()
                        self.status.set('Session complete. Review it, then select the next session and reposition your hand.' if not all(s['complete'] for s in self.batch.data['sessions']) else 'All three sessions complete. Review, then Process and save.')
                if self.recording:
                    quality = 'RECORDING · ' + quality
                self.quality.set(quality)
                height, width = annotated.shape[:2]
                scale = min(max(1, self.preview.winfo_width()) / width,
                            max(1, self.preview.winfo_height()) / height)
                image = cv2.resize(annotated, (max(1, int(width * scale)), max(1, int(height * scale))))
                self.photo = tk.PhotoImage(data=cv2.imencode('.png', image)[1].tobytes())
                self.preview.configure(image=self.photo)
            except Exception as error:
                self.close_camera()
                self.status.set(f'Capture paused: {error}. Retry the camera or discard the batch.')
        self.root.after(60, self.tick)

    def discard(self):
        if self.batch and messagebox.askyesno('Discard batch', 'Delete all three sessions in this unfinished batch? Existing dataset entries and backups will be preserved.'):
            self.recording = False
            self.countdown = 0
            self.batch.discard()
            self.batch = None
            self.update_progress()
            self.controls()

    def process(self):
        if not self.batch or self.busy:
            return
        replacing = self.mode.get() == 'Replace this sign'
        if replacing and not messagebox.askyesno('Replace sign', f"Replace all existing {self.batch.data['label']} rows after backup? Other labels will remain."):
            return
        self.close_camera()
        self.busy = True
        self.controls()
        batch, target = self.batch, self.csv.get()
        def work():
            try:
                result = process_batch(batch, target, replace=replacing, progress=lambda value: self.events.put(('log', value)))
                self.events.put(('processed', result))
            except Exception as error:
                self.events.put(('error', str(error)))
        threading.Thread(target=work, daemon=True).start()
        self.status.set('Extracting recordings. Working files are retained if processing fails.')

    def train(self):
        if self.busy or self.batch:
            return
        try:
            epochs, size = self.epochs.get(), self.batch_size.get()
            if epochs < 1 or size < 1:
                raise ValueError('Epochs and batch size must be positive.')
            self.close_camera()
            run = PROJECT / 'ml/models/manual_runs' / identifier()
            run.mkdir(parents=True)
            snapshot = run / 'landmarks.csv'
            output = run / 'sign_classifier.pth'
            source = self.csv.get()
            self.train_stop.clear()
            self.busy = True
            self.train_bar['value'] = 0
            self.training_progress.set('Checking dataset...')
            self.controls()
            self.status.set('Preparing and checking an isolated dataset snapshot...')
            self.write(f'Training output: {run}')
            def monitor():
                try:
                    from ml.data_io import read_samples
                    from ml.dataset import create_splits, validate_splits
                    from ml.preprocessing import STATIC_HAND_CANONICAL
                    from ml.manual_recording import file_hash
                    before = file_hash(Path(source))
                    shutil.copyfile(source, snapshot)
                    if file_hash(snapshot) != before or file_hash(Path(source)) != before:
                        raise ValueError('Dataset changed while preparing the training snapshot. Retry training.')
                    samples = read_samples(snapshot)
                    splits = create_splits(samples)
                    validate_splits(samples, splits, static_hand_preprocessing=STATIC_HAND_CANONICAL)
                    if self.train_stop.is_set():
                        self.events.put(('trained', (-1, str(output))))
                        return
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'ml.train', '--data', str(snapshot), '--output', str(output), '--epochs', str(epochs), '--batch-size', str(size)], cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    self.events.put(('training_started', process))
                    with (run / 'training.log').open('w', encoding='utf-8') as log:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            self.events.put(('log', line))
                    self.events.put(('trained', (process.wait(), str(output))))
                except Exception as error:
                    self.events.put(('training_error', str(error)))
            threading.Thread(target=monitor, daemon=True).start()
        except Exception as error:
            messagebox.showerror('Cannot train', str(error))

    def stop_training(self):
        self.train_stop.set()
        if self.training and self.training.poll() is None:
            self.training.terminate()
            self.status.set('Stopping training. Any partial model stays in its version folder; existing models are preserved.')

    def choose_model(self):
        if self.busy or self.batch:
            return
        path = filedialog.askopenfilename(initialdir=PROJECT / 'ml/models', filetypes=[('PyTorch model', '*.pth')])
        if path:
            try:
                from ml.predict import SignPredictor
                SignPredictor(path)
                self.model = path
                self.status.set(f'Selected static model: {Path(path).name}')
                self.controls()
            except Exception as error:
                messagebox.showerror('Cannot use static model', str(error))

    def test(self):
        if self.busy or self.batch or not self.model:
            return
        if self.test_process and self.test_process.poll() is None:
            self.status.set('Camera test is already open. Press Q in its window to finish.')
            return
        self.close_camera()
        try:
            folder = self.workspace.root / 'test-logs'
            folder.mkdir(exist_ok=True)
            self.test_log_path = folder / (identifier() + '.log')
            self.test_log = self.test_log_path.open('w', encoding='utf-8')
            self.test_process = subprocess.Popen([sys.executable, '-u', '-m', 'ml.predict', '--sign-model', self.model, '--camera', str(self.camera_index.get())], cwd=PROJECT, stdout=self.test_log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.status.set('Camera test opened. Press Q in its window to finish. The test does not change the active deployment model.')
        except Exception as error:
            if self.test_log:
                self.test_log.close()
                self.test_log = None
            messagebox.showerror('Cannot open model test', str(error))

    def poll(self):
        if self.test_process and self.test_process.poll() is not None:
            code = self.test_process.returncode
            self.test_process = None
            if self.test_log:
                self.test_log.close()
                self.test_log = None
            self.write(f'Camera test ended (exit {code}). Log: {self.test_log_path}')
            if code != 0:
                self.status.set(f'Camera test failed. Check the log: {self.test_log_path}')
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log':
                    self.write(value)
                    if str(value).startswith('Epoch '):
                        epoch, total = str(value).split()[1].split('/')
                        self.train_bar['value'] = int(epoch) / int(total) * 100
                        self.training_progress.set(f'Epoch {epoch} / {total}')
                elif kind == 'processed':
                    self.batch = None
                    self.busy = False
                    self.write(f"Saved {value['samples']} samples. Recording backup: {value['archive']}")
                    self.status.set('Dataset saved. Working recordings removed; archive preserved.')
                    self.update_progress()
                    self.refresh()
                elif kind == 'error':
                    self.busy = False
                    self.status.set('Processing failed. Retake a session or retry; working recordings were retained.')
                    self.write(value)
                    if not self.closing:
                        messagebox.showerror('Processing failed', value)
                elif kind == 'trained':
                    self.busy = False
                    self.training = None
                    code, model = value
                    if code == 0:
                        self.model = model
                        self.training_progress.set('Training complete')
                        self.status.set('Training complete. Test the new model with the camera.')
                    else:
                        self.training_progress.set('Stopped or failed')
                        self.status.set('Training stopped or failed. Review the log. Existing models were preserved.')
                elif kind == 'training_started':
                    self.training = value
                    if self.train_stop.is_set():
                        self.stop_training()
                    else:
                        self.status.set('Training in progress. Epoch results appear below.')
                elif kind == 'training_error':
                    self.busy = False
                    self.training = None
                    self.status.set('Training could not start. Review the log.')
                    self.training_progress.set('Dataset check or training failed')
                    self.write(value)
                self.controls()
        except queue.Empty:
            pass
        if self.closing and not self.busy:
            self.finish_close()
            return
        self.root.after(100, self.poll)

    def close(self):
        if self.batch and not messagebox.askyesno('Exit recording', 'Exiting discards all working recordings in this batch. Completed dataset entries and backups remain. Exit?'):
            return
        if self.training and not messagebox.askyesno('Stop training', 'Stop this training run and exit? Existing models will remain intact.'):
            return
        self.closing = True
        self.close_camera()
        self.stop_training()
        if self.busy:
            self.status.set('Finishing the current dataset transaction before closing...')
        else:
            self.finish_close()

    def finish_close(self):
        if self.batch:
            self.batch.discard()
        if self.test_process and self.test_process.poll() is None:
            self.test_process.terminate()
        if self.test_log:
            self.test_log.close()
        self.workspace.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ManualTrainingApp(root)
    except Exception as error:
        messagebox.showerror('Cannot open recording studio', str(error))
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    main()
