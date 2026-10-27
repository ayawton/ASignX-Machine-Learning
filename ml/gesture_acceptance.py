"""Motion evidence for completed takes; this does not verify ASL correctness."""
from dataclasses import dataclass
import numpy as np


def is_negative_label(label: str) -> bool:
    return label.strip().casefold().replace('-', '_').replace(' ', '_') in {
        'background', 'idle', 'no_gesture',
    }


@dataclass(frozen=True)
class MotionAcceptance:
    # Initial engineering defaults, not thresholds fitted to human negatives.
    min_excursion: float = .35  # palm widths
    min_motion_seconds: float = .12
    min_speed: float = .3  # palm widths / second

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.min_excursion, self.min_motion_seconds, self.min_speed)):
            raise ValueError('Motion acceptance thresholds must be positive and finite')

    def validate(self, frames):
        """Require one continuous hand track to move substantially over time.

        Work in image x/y, scale by palm width, and median-filter three
        consecutive observations to suppress one-frame spikes. Never join
        tracks across missing frames. Depth jitter cannot count as motion.
        """
        tracks = {name: [] for _, hands in frames for name in hands}
        for stamp, hands in frames:
            for name in tracks:
                tracks[name].append((stamp, hands.get(name)))
        best_excursion = best_duration = 0.
        for entries in tracks.values():
            segments, current = [], []
            for stamp, points in entries:
                if points is None:
                    segments.append(current); current = []
                    continue
                if current and stamp-current[-1][0] > .2:
                    segments.append(current); current = []
                current.append((stamp, points))
            segments.append(current)
            for segment in segments:
                if len(segment) < 5:
                    continue
                stamps = np.array([s for s, _ in segment])
                points = np.stack([p for _, p in segment])[:, :, :2]
                scale = float(np.median(np.linalg.norm(points[:, 5]-points[:, 17], axis=1)))
                if scale <= 1e-5:
                    continue
                tracked = points[:, [0, 4, 8, 12, 16, 20]]
                smooth = np.median(np.stack([tracked[:-2], tracked[1:-1], tracked[2:]]), axis=0)
                stamps = stamps[1:-1]
                excursion = float(np.linalg.norm(
                    np.quantile(smooth, .95, axis=0)-np.quantile(smooth, .05, axis=0), axis=1).max()/scale)
                dt = np.diff(stamps)
                speed = np.linalg.norm(np.diff(smooth, axis=0), axis=2).max(axis=1)/scale/dt
                duration = float(dt[speed >= self.min_speed].sum())
                best_excursion = max(best_excursion, excursion)
                best_duration = max(best_duration, duration)
                if excursion >= self.min_excursion and duration >= self.min_motion_seconds:
                    return
        raise ValueError(f'Insufficient sustained motion ({best_excursion:.2f} palm widths, '
                         f'{best_duration:.2f}s moving)')
