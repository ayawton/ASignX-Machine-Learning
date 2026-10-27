"""Opt-in timing/tracking logs; no camera images or landmark coordinates."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import numpy as np


class LiveDiagnostics:
    TARGET_KEYS = {ord('7'): 'J', ord('8'): 'Z', ord('9'): 'HELLO', ord('0'): 'REST'}

    def __init__(self, output, metadata=None):
        self.root = Path(output) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6])
        self.root.mkdir(parents=True, exist_ok=False)
        self.stream = (self.root/'frames.jsonl').open('w', encoding='utf-8')
        self.metadata = metadata or {}
        self.target = 'UNMARKED'
        self.rows = []
        self.events = []
        self.closed = False

    def mark(self, key):
        if key not in self.TARGET_KEYS:
            return False
        self.target = self.TARGET_KEYS[key]
        return True

    def record(self, result, info, *, capture_time, processed_time, detector_ms, recognition_ms):
        info = info or {}
        body = getattr(result, 'body_context', None)
        shoulders = face = None
        if body is not None:
            body = np.asarray(body)
            if body.ndim == 2 and body.shape[0] >= 8:
                shoulders = bool(np.isfinite(body[2:4]).all())
                face = bool(np.isfinite(body[6:8]).all())
        prediction = info.get('dynamic_prediction')
        trace_file = None
        if info.get('completed_take') is not None:
            trace_file = self.record_take(info['completed_take'], info)
        row = {
            'frame': len(self.rows), 'target': self.target,
            'capture_time': float(capture_time), 'processed_time': float(processed_time),
            'frame_age_ms': max(0., float(processed_time-capture_time)*1000),
            'detector_ms': float(detector_ms), 'recognition_ms': float(recognition_ms),
            'hand_visible': bool(result.all_hands),
            'left_visible': any(h.handedness == 'Left' for h in result.all_hands),
            'right_visible': any(h.handedness == 'Right' for h in result.all_hands),
            'shoulders_visible': shoulders, 'face_references_visible': face,
            'state': info.get('state'), 'velocity': float(info.get('velocity', 0)),
            'emitted': info.get('dynamic_emitted'), 'rejected': info.get('dynamic_rejected'),
            'evaluated': bool(info.get('dynamic_evaluated')),
            'raw_prediction': prediction.raw_label if prediction else None,
            'confidence': float(prediction.confidence) if prediction else None,
            'static_candidates': [r.prediction.raw_label for r in info.get('static', [])],
            'static_emitted': [r.emitted_label for r in info.get('static', []) if r.emitted_label],
            'static_suppressed': any(getattr(r, 'suppressed', False) for r in info.get('static', [])),
            'static_awaiting_release': any(getattr(r,'awaiting_release',False) for r in info.get('static',[])),
            'gesture_input_file': trace_file,
        }
        if row['emitted'] or row['rejected'] or row['evaluated'] or trace_file:
            self.events.append(row.copy())
        self.rows.append(row)
        self.stream.write(json.dumps(row)+'\n')
        if len(self.rows)%30 == 0:
            self.stream.flush()

    def record_take(self, frames, info):
        """Opt-in replay data, saved separately from normal metric-only logs."""
        if not frames:
            return None
        shape = next((np.asarray(frame).shape for _,frame in frames if frame is not None), None)
        if shape is None:
            return None
        presence = np.array([frame is not None for _,frame in frames], dtype=bool)
        values = np.stack([np.asarray(frame,dtype=np.float32) if frame is not None
                           else np.full(shape,np.nan,dtype=np.float32) for _,frame in frames])
        prediction = info.get('dynamic_prediction')
        relative = Path('gesture_inputs') / f'take_{len(self.rows):06d}.npz'
        (self.root/relative).parent.mkdir(exist_ok=True)
        metadata = {'target':self.target,'settings':self.metadata,
                    'emitted':info.get('dynamic_emitted'),'rejected':info.get('dynamic_rejected'),
                    'raw_prediction':prediction.raw_label if prediction else None,
                    'confidence':float(prediction.confidence) if prediction else None,
                    'purpose':'Diagnostic replay only; target is user-marked, not a reviewed training label'}
        motion=info.get('completed_motion_take')
        extra={}
        if motion:
            extra={'motion_timestamps':np.array([t for t,_ in motion],dtype=np.float64),
                   'motion_hands':np.array([[hands.get(name,np.full((21,3),np.nan))
                                            for name in ('Left','Right')] for _,hands in motion],dtype=np.float32)}
        np.savez_compressed(self.root/relative,
                            timestamps=np.array([stamp for stamp,_ in frames],dtype=np.float64),
                            frames=values,present=presence,metadata=json.dumps(metadata),**extra)
        return relative.as_posix()

    @staticmethod
    def describe(rows):
        if not rows:
            return {'frames': 0}
        # Repeated test blocks can be separated by other labels. Do not count
        # time spent testing Z as slow processing in the J block, for example.
        adjacent = [(a,b) for a,b in zip(rows,rows[1:]) if b['frame']==a['frame']+1]
        span = sum(b['processed_time']-a['processed_time'] for a,b in adjacent)
        intervals = np.array([b['capture_time']-a['capture_time'] for a,b in adjacent])
        def visibility(key):
            values = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None
        return {
            'frames': len(rows),
            'processed_fps': len(adjacent)/span if span > 0 else None,
            'detector_ms_median': float(np.median([r['detector_ms'] for r in rows])),
            'detector_ms_p95': float(np.quantile([r['detector_ms'] for r in rows], .95)),
            'recognition_ms_p95': float(np.quantile([r['recognition_ms'] for r in rows], .95)),
            'frame_age_ms_p95': float(np.quantile([r['frame_age_ms'] for r in rows], .95)),
            'capture_interval_ms_p95': float(np.quantile(intervals, .95)*1000) if len(intervals) else None,
            'capture_intervals_over_200ms': int((intervals > .2).sum()),
            'hand_visibility': visibility('hand_visible'),
            'left_visibility': visibility('left_visible'), 'right_visibility': visibility('right_visible'),
            'shoulder_visibility': visibility('shoulders_visible'),
            'face_reference_visibility': visibility('face_references_visible'),
            'emissions': dict(Counter(r['emitted'] for r in rows if r['emitted'])),
            'static_emissions': dict(Counter(label for r in rows for label in r.get('static_emitted', []))),
            'rejections': dict(Counter(r['rejected'] for r in rows if r['rejected'])),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stream.close()
        summary = {
            'metadata': self.metadata,
            'overall': self.describe(self.rows),
            'by_target': {target: self.describe([r for r in self.rows if r['target'] == target])
                          for target in sorted({r['target'] for r in self.rows})},
            'events': self.events,
            'limitations': [
                'Key labels describe the intended test block, not verified ASL annotations.',
                'Frame visibility includes rest between gestures; gesture-specific rejection checks use sampled takes.',
                'Processed FPS describes observed pipeline throughput, not the camera advertised FPS.',
                'Timing cannot by itself establish motion blur or ASL correctness.',
            ],
        }
        (self.root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
