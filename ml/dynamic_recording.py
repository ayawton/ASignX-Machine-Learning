"""Camera-independent controls for the dynamic sign studio."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from ml.collect_sequence import _save_take
from ml.sequence_io import load_sequence_file
from ml.video import HAND_SELECTIONS, LandmarkSequenceCapture


@dataclass(frozen=True)
class CaptureSettings:
    output: Path
    label: str
    hand: str = "both"
    sequence_length: int = 30
    target_fps: float = 15.0
    min_frames: int = 10
    min_detection_ratio: float = 0.5
    max_seconds: float = 10.0
    countdown_seconds: float = 3.0
    signer_id: str = ""
    recording_id: str = ""
    source_id: str = ""
    device: str = ""
    split: str = ""
    body_context: bool = False
    temple_context: bool = False

    def validate(self) -> None:
        if self.split not in ("", "train", "validation", "test"):
            raise ValueError("Reserved split must be train, validation or test.")
        if not self.label.strip():
            raise ValueError("Enter a sign label, such as J, Z, or HELLO.")
        if self.hand not in HAND_SELECTIONS:
            raise ValueError("Choose a valid hand mode.")
        if self.sequence_length < 2 or self.min_frames < 2:
            raise ValueError("Sequence length and minimum frames must be at least 2.")
        if not math.isfinite(self.target_fps) or not 0 < self.target_fps <= 120:
            raise ValueError("Sampling FPS must be between 0 and 120.")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 120:
            raise ValueError("Maximum take duration must be between 0 and 120 seconds.")
        if not 0 < self.min_detection_ratio <= 1:
            raise ValueError("Minimum detection ratio must be greater than 0 and at most 1.")
        if not math.isfinite(self.countdown_seconds) or not 0 <= self.countdown_seconds <= 30:
            raise ValueError("Preparation countdown must be between 0 and 30 seconds.")
        if self.max_seconds * self.target_fps < self.min_frames:
            raise ValueError("Increase maximum duration or FPS to allow the minimum frames.")
        if (self.body_context or self.temple_context) and self.hand != "both":
            raise ValueError("Body context requires hand mode both.")
        validate_output(self.output, self.hand, self.sequence_length, self.body_context, self.temple_context)


def validate_output(output: Path, hand: str, sequence_length: int, body_context: bool = False, temple_context: bool = False) -> None:
    """Prevent incompatible hand contracts from being mixed, even at equal widths."""
    width = 180 if temple_context else 150 if body_context else 126 if hand == "both" else 63
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a folder, not a file.")
    for path in output.rglob("*.npz"):
        sample = load_sequence_file(path)
        if (sample.features.shape != (sequence_length, width)
                or sample.metadata.get("hand_selection") != hand):
            raise ValueError(
                f"{path.name} uses different hand settings or sequence length. "
                "Choose matching settings or a new output folder."
            )


class TakeRecorder:
    """One active countdown/take; the GUI worker owns this object exclusively."""

    def __init__(self, settings: CaptureSettings):
        settings.validate()
        self.settings = settings
        self.capture: LandmarkSequenceCapture | None = None
        self.deadline: float | None = None
        self.started_at = 0.0
        self.next_sample_at = 0.0
        self.saved: list[Path] = []

    @property
    def state(self) -> str:
        return "Recording" if self.capture is not None else "Preparing" if self.deadline is not None else "Ready"

    def start(self, now: float) -> None:
        if self.state != "Ready":
            raise ValueError("Finish or discard the current take first.")
        self.deadline = now + self.settings.countdown_seconds

    def discard(self) -> None:
        self.capture = None
        self.deadline = None

    def update(self, hands, now: float, body_context=None) -> Path | None:
        if self.deadline is not None and now >= self.deadline:
            self.deadline = None
            self.capture = LandmarkSequenceCapture(self.settings.hand, body_context=self.settings.body_context, temple_context=self.settings.temple_context)
            self.started_at = self.next_sample_at = now
        if self.capture is not None:
            if now >= self.next_sample_at:
                self.capture.add_hands(hands, body_context)
                self.next_sample_at = now + 1.0 / self.settings.target_fps
            if now - self.started_at >= self.settings.max_seconds:
                return self.save(now)
        return None

    def save(self, now: float) -> Path:
        if self.capture is None:
            raise ValueError("Wait for the countdown and perform the sign before saving.")
        capture = self.capture
        self.discard()
        path = _save_take(
            capture, label=self.settings.label.strip(), output_dir=self.settings.output,
            sequence_length=self.settings.sequence_length, target_fps=self.settings.target_fps,
            min_frames=self.settings.min_frames, min_detection_ratio=self.settings.min_detection_ratio,
            duration_seconds=now - self.started_at,
            provenance={key: getattr(self.settings, key) for key in
                        ("signer_id", "recording_id", "source_id", "device", "split")
                        if getattr(self.settings, key)},
        )
        self.saved.append(path)
        return path

    def describe(self, now: float) -> str:
        if self.deadline is not None:
            return f"Get ready: {max(0, math.ceil(self.deadline - now))}"
        if self.capture is not None:
            return (f"RECORDING {now - self.started_at:.1f}s | "
                    f"{self.capture.frame_count} frames | hand visible {self.capture.detection_ratio:.0%}"
                    + (f" | temples visible {self.capture.temple_detection_ratio:.0%}" if self.settings.temple_context else ""))
        return f"Ready | {len(self.saved)} new take(s) saved"
