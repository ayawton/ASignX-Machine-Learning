"""Isolated three-session static capture and recoverable dataset transactions."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import zipfile

import numpy as np

from ml.data_io import CSV_COLUMNS, read_samples
from ml.preprocessing import (
    PREPROCESSING_VERSION, canonicalize_static_features,
    normalize_landmarks,
)


def identifier() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12]


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RecordingWorkspace:
    """An OS-held lock prevents one GUI cleaning another GUI's active recordings."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'gui.lock').open('a+b')
        self.lock.seek(0)
        self.lock.write(b'0')
        self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError('Another recording GUI is using this workspace.') from None
        self.pending = self.root / 'pending'
        self.pending.mkdir(exist_ok=True)
        self.backups = self.root / 'backups'
        self.backups.mkdir(exist_ok=True)

    def cleanup_abandoned(self) -> int:
        count = 0
        for path in self.pending.iterdir():
            if path.is_dir() and not path.is_symlink():
                self.remove_batch(path)
                count += 1
        return count

    def remove_batch(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.parent != self.pending.resolve() or path.is_symlink():
            raise ValueError('Refusing to remove a path outside the recording workspace.')
        shutil.rmtree(resolved)

    def close(self) -> None:
        self.lock.close()


class RecordingBatch:
    def __init__(self, workspace: RecordingWorkspace, label: str, hand: str,
                 samples: int = 200, camera: int = 0):
        label = label.strip().upper()
        if not label or len(label) > 80 or any(ord(c) < 32 for c in label):
            raise ValueError('Enter a sign label of 1 to 80 printable characters.')
        if hand not in ('Left', 'Right') or samples < 10:
            raise ValueError('Choose Left or Right and at least 10 samples per session.')
        self.workspace = workspace
        self.path = workspace.pending / identifier()
        self.path.mkdir()
        self.data = dict(batch_id=self.path.name, label=label, hand=hand,
                         samples=samples, camera=camera, state='recording',
                         capture_settings=dict(interval_seconds=0.15, exposure='auto',
                                               movement_limit_per_frame=0.025),
                         preprocessing=PREPROCESSING_VERSION,
                         sessions=[dict(id=uuid.uuid4().hex, complete=False, frames=0)
                                   for _ in range(3)])
        self.save()

    def save(self) -> None:
        atomic_json(self.path / 'manifest.json', self.data)

    def session_path(self, index: int) -> Path:
        if index not in range(3):
            raise ValueError('Session must be 1, 2 or 3.')
        return self.path / self.data['sessions'][index]['id']

    def add_frame(self, index: int, frame: np.ndarray) -> int:
        import cv2
        session = self.data['sessions'][index]
        if session['complete'] or session['frames'] >= self.data['samples']:
            raise ValueError('Session is already recorded. Retake it to make changes.')
        folder = self.session_path(index)
        folder.mkdir(exist_ok=True)
        number = session['frames'] + 1
        if not cv2.imwrite(str(folder / f'{number:05d}.png'), frame):
            raise OSError('Could not save recording frame. Check free disk space.')
        session['frames'] = number
        self.save()
        return number

    def finish_session(self, index: int) -> None:
        session = self.data['sessions'][index]
        if session['frames'] != self.data['samples']:
            raise ValueError('Finish the required sample count before accepting this session.')
        session['complete'] = True
        self.save()

    def retake(self, index: int) -> None:
        folder = self.session_path(index)
        if folder.exists():
            # The fixed session UUID is always a direct child of this batch.
            if folder.resolve().parent != self.path.resolve():
                raise ValueError('Invalid session path.')
            shutil.rmtree(folder.resolve())
        self.data['sessions'][index].update(complete=False, frames=0)
        self.save()

    def discard(self) -> None:
        self.workspace.remove_batch(self.path)


def dataset_overview(path: str | Path) -> list[tuple[str, int, int]]:
    path = Path(path)
    if not path.exists():
        return []
    counts = Counter()
    sessions = defaultdict(set)
    for sample in read_samples(path):
        counts[sample.label] += 1
        sessions[sample.label].add(sample.session_id)
    return [(label, counts[label], len(sessions[label])) for label in sorted(counts)]


def process_batch(batch: RecordingBatch, csv_path: str | Path, *, replace: bool = False,
                  detector_factory=None, progress=lambda message: None) -> dict:
    """Extract first, archive and verify, then atomically publish the validated CSV.

    Failure before publication leaves the original CSV and working recordings intact.
    Session IDs make retrying publication idempotent after an interrupted cleanup.
    """
    import cv2
    if not all(s['complete'] for s in batch.data['sessions']):
        raise ValueError('Complete all three sessions before processing.')
    target = Path(csv_path).resolve()
    if target.is_relative_to(batch.workspace.root):
        raise ValueError('The dataset must be outside the temporary recording workspace.')
    target.parent.mkdir(parents=True, exist_ok=True)
    before = file_hash(target)
    existing = read_samples(target) if target.exists() else []
    own_sessions = {s['id'] for s in batch.data['sessions']}
    published = [s for s in existing if s.session_id in own_sessions]
    if published:
        archive = batch.workspace.backups / (batch.path.name + '.zip')
        if {s.session_id for s in published} != own_sessions or not archive.exists():
            raise ValueError('Partial prior publication detected; restore the dataset backup before retrying.')
        with zipfile.ZipFile(archive) as check:
            if check.testzip():
                raise ValueError('Recording archive is damaged; working recordings were retained.')
        batch.discard()
        return dict(samples=len(published), archive=str(archive), csv=str(target), recovered=True)
    if detector_factory is None:
        from asignx import HandDetector
        detector_factory = lambda: HandDetector(static_image_mode=True, max_num_hands=2)
    kept = [s for s in existing if not (replace and s.label == batch.data['label'])]
    fingerprints = {}
    for sample in kept:
        key = canonicalize_static_features(sample.features, sample.handedness).tobytes()
        fingerprints[key] = sample.label
    rows = []
    accepted = []
    detector = detector_factory()
    try:
        for number, session in enumerate(batch.data['sessions'], 1):
            count = 0
            for image in sorted(batch.session_path(number - 1).glob('*.png')):
                frame = cv2.imread(str(image))
                if frame is None:
                    raise ValueError(f'Unreadable recording: {image.name}')
                result, _ = detector.process(frame, draw=False)
                hand = next((h for h in result.all_hands if h.handedness == batch.data['hand']), None)
                if hand is None:
                    continue
                try:
                    features = normalize_landmarks(hand.to_numpy())
                    key = canonicalize_static_features(features, hand.handedness).tobytes()
                except ValueError:
                    continue
                if key in fingerprints:
                    if fingerprints[key] != batch.data['label']:
                        raise ValueError('A recorded pose exactly matches another sign label. Review the recordings.')
                    continue
                fingerprints[key] = batch.data['label']
                rows.append([batch.data['label'], session['id'], hand.handedness,
                             PREPROCESSING_VERSION, *features.tolist()])
                count += 1
            minimum = max(10, int(np.ceil(batch.data['samples'] * 0.8)))
            if count < minimum:
                raise ValueError(f'Session {number}: only {count} unique valid samples; need {minimum}. Retake this session.')
            accepted.append(count)
            progress(f'Extracted session {number}/3: {count} valid samples')
    finally:
        detector.close()
    staged = batch.path / 'landmarks.csv'
    with staged.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for sample in kept:
            writer.writerow([sample.label, sample.session_id, sample.handedness,
                             PREPROCESSING_VERSION, *sample.features.tolist()])
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    read_samples(staged)
    batch.data.update(state='processed', accepted=accepted, destination=str(target), replace=replace)
    batch.save()
    archive = batch.workspace.backups / (batch.path.name + '.zip')
    temporary_archive = archive.with_suffix('.zip.tmp')
    progress('Backing up and verifying original recordings...')
    with zipfile.ZipFile(temporary_archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in batch.path.rglob('*'):
            if path.is_file():
                output.write(path, path.relative_to(batch.path).as_posix())
    with zipfile.ZipFile(temporary_archive) as check:
        if check.testzip():
            raise OSError('Recording archive verification failed.')
    os.replace(temporary_archive, archive)
    publish = target.with_name(target.name + '.' + batch.path.name + '.tmp')
    try:
        shutil.copyfile(staged, publish)
        with publish.open('r+b') as stream:
            os.fsync(stream.fileno())
        if file_hash(target) != before:
            raise ValueError('Dataset changed during processing. Retry using the current dataset.')
        csv_backup = None
        if target.exists():
            csv_backup = target.with_name(target.name + '.backup-' + batch.path.name)
            shutil.copyfile(target, csv_backup)
            if file_hash(csv_backup) != before or file_hash(target) != before:
                raise ValueError('Dataset changed during backup. Retry processing.')
        os.replace(publish, target)
    finally:
        publish.unlink(missing_ok=True)
    result = dict(samples=len(rows), archive=str(archive), csv=str(target),
                  csv_backup=str(csv_backup) if csv_backup else None)
    batch.discard()
    return result
