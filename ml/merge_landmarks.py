"""Preview and safely merge portable static datasets, preserving primary CSV bytes."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np

from ml.data_io import CSV_COLUMNS
from ml.preprocessing import FEATURE_COLUMNS, PREPROCESSING_VERSION, validate_features


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class MergePreview:
    main: Path
    incoming: Path
    main_hash: str
    incoming_hash: str
    main_rows: int
    accepted: list[list[str]]
    decisions: list[dict]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(item['decision'] for item in self.decisions))


def _read(data: bytes):
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''), strict=True)
    if next(reader, None) != CSV_COLUMNS:
        raise ValueError('Incompatible CSV header. Both files must use the current 67-column landmarks format.')
    return list(reader)


def _validate(row: list[str]):
    if len(row) != len(CSV_COLUMNS):
        raise ValueError('Wrong number of columns')
    label, session, hand, version = row[:4]
    if not label.strip() or not session.strip():
        raise ValueError('Missing label or session ID')
    if version != PREPROCESSING_VERSION:
        raise ValueError('Incompatible preprocessing version')
    if hand.strip() not in ('', 'Left', 'Right'):
        raise ValueError('Handedness must be Left, Right or empty')
    features = validate_features(np.asarray(row[4:], dtype=np.float32)).copy()
    features[features == 0] = 0  # Treat signed zero as identical.
    return label.strip(), session.strip(), features.tobytes()


def preview_merge(main: str | Path, incoming: str | Path) -> MergePreview:
    main, incoming = Path(main).resolve(), Path(incoming).resolve()
    if main == incoming or os.path.samefile(main, incoming):
        raise ValueError('Select two different CSV files.')
    main_data, incoming_data = main.read_bytes(), incoming.read_bytes()
    main_rows, incoming_rows = _read(main_data), _read(incoming_data)
    sessions: dict[str, set[str]] = defaultdict(set)
    existing: dict[bytes, set[str]] = defaultdict(set)
    for number, row in enumerate(main_rows, 2):
        try:
            label, session, key = _validate(row)
            sessions[session].add(label)
            if len(sessions[session]) > 1:
                raise ValueError('Session ID belongs to multiple labels')
            existing[key].add(label)
        except ValueError as exc:
            raise ValueError(f'Main CSV row {number}: {exc}. Main CSV was not changed.') from exc

    valid = []
    decisions = []
    incoming_labels: dict[bytes, set[str]] = defaultdict(set)
    incoming_sessions: dict[str, set[str]] = defaultdict(set)
    for number, row in enumerate(incoming_rows, 2):
        # Different preprocessing requires re-extraction, not silent partial import.
        if len(row) == len(CSV_COLUMNS) and row[3] != PREPROCESSING_VERSION:
            raise ValueError(f'Incoming CSV row {number}: incompatible preprocessing version.')
        try:
            label, session, key = _validate(row)
            valid.append((number, row, label, session, key))
            incoming_labels[key].add(label)
            incoming_sessions[session].add(label)
        except ValueError as exc:
            decisions.append(dict(row=number, label=row[0] if row else '', decision='invalid', reason=str(exc)))

    accepted = []
    seen = set()
    for number, row, label, session, key in valid:
        if key in existing:
            decision = 'duplicate' if existing[key] == {label} else 'label conflict'
        elif len(incoming_labels[key]) > 1:
            decision = 'label conflict'
        elif (session in sessions and sessions[session] != {label}) or len(incoming_sessions[session]) > 1:
            decision = 'session conflict'
        elif key in seen:
            decision = 'duplicate'
        else:
            decision = 'append'
            accepted.append(row)
            seen.add(key)
        decisions.append(dict(row=number, label=label, session_id=session, decision=decision))
    decisions.sort(key=lambda item: item['row'])
    return MergePreview(main, incoming, digest(main_data), digest(incoming_data),
                        len(main_rows), accepted, decisions)


def commit_merge(preview: MergePreview) -> tuple[Path, Path]:
    """Commit a reviewed snapshot; never merge a changed input or delete another writer's lock."""
    lock = preview.main.with_name(preview.main.name + '.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f'Dataset is locked: {lock}. Close other writers; do not remove a lock while extraction is running.') from exc
    os.close(fd)
    temporary = None
    try:
        # Recompute instead of trusting mutable preview rows supplied by a caller.
        current = preview_merge(preview.main, preview.incoming)
        if (current.main_hash, current.incoming_hash) != (preview.main_hash, preview.incoming_hash):
            raise ValueError('A CSV changed after preview. Preview again before merging.')
        original = current.main.read_bytes()
        if digest(original) != current.main_hash:
            raise ValueError('Main CSV changed during merge. Preview again.')
        token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
        backup = current.main.with_name(current.main.name + f'.backup-{token}')
        report = current.main.with_name(current.main.name + f'.merge-{token}.json')
        fd, name = tempfile.mkstemp(prefix='.merge-', suffix='.csv', dir=current.main.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(original)
            if current.accepted:
                if original and not original.endswith((b'\r', b'\n')):
                    handle.write(b'\n')
                buffer = io.StringIO(newline='')
                csv.writer(buffer).writerows(current.accepted)
                handle.write(buffer.getvalue().encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
        output_hash = digest(temporary.read_bytes())
        with backup.open('xb') as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        # Persist the audit before replacing data. Hashes identify whether replacement completed.
        with report.open('x', encoding='utf-8') as handle:
            json.dump(dict(main=str(current.main), incoming=str(current.incoming), backup=str(backup),
                           main_before_sha256=current.main_hash, incoming_sha256=current.incoming_hash,
                           expected_output_sha256=output_hash, main_rows=current.main_rows,
                           counts=current.counts, decisions=current.decisions,
                           note='Prepared merge audit. Compare main CSV hash with expected_output_sha256 to verify application.'),
                      handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if (digest(current.main.read_bytes()) != current.main_hash or
                digest(current.incoming.read_bytes()) != current.incoming_hash):
            raise ValueError('A CSV changed during merge. No replacement made; preview again.')
        if current.accepted:
            os.replace(temporary, current.main)
        return backup, report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
