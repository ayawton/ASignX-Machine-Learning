"""
ASignX Hand Recognition Module - Visual Debugger & Overlay.

Provides HUD overlay, 1-hand / 2-hand status indicators, and coordinate
readouts for visual verification during development.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from asignx.models import HandDetectionResult


class VisualDebugger:
    """Renders diagnostic information, status banners, and landmark readouts on frames."""

    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_landmarks: bool = True
        self.show_hud: bool = True

    def render_overlay(
        self,
        frame: np.ndarray,
        result: HandDetectionResult,
        fps: float = 0.0,
        recognitions: list | None = None,
        dynamic_info: dict | None = None,
    ) -> np.ndarray:
        """Render debug overlay and real-time sign recognition HUD over the given frame.

        Args:
            frame: Base frame (already containing landmark skeleton if drawn).
            result: Hand detection result data.
            fps: Current processing frames-per-second.
            recognitions: Optional list of static Recognition objects from SignRecognizer.
            dynamic_info: Optional dict from UnifiedSignRecognizer for dynamic gestures.

        Returns:
            Frame with HUD elements rendered.
        """
        if not self.show_hud:
            return frame

        canvas = frame.copy()
        height, width, _ = canvas.shape

        # 1. Top Status Banner (Semi-transparent background)
        header_h = 70
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, header_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # Title and FPS
        title_text = "ASignX Hand Recognition Module"
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            canvas, title_text, (16, 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        fps_color = (0, 255, 128) if fps >= 28.0 else ((0, 255, 255) if fps >= 18.0 else (0, 100, 255))
        cv2.putText(
            canvas, fps_text, (width - 150, 30), self.font, 0.7, fps_color, 2, cv2.LINE_AA
        )

        # Detection Status Banner
        if result.hand_detected and result.all_hands:
            hands_count = len(result.all_hands)
            if hands_count >= 2:
                labels = ", ".join([h.handedness or f"Hand{i+1}" for i, h in enumerate(result.all_hands)])
                total_lms = sum(len(h) for h in result.all_hands)
                status_text = f"STATUS: DETECTED (2 Hands: {labels}) | Landmarks: {total_lms}/42"
            else:
                h = result.all_hands[0]
                hand_label = h.handedness or "Hand"
                status_text = f"STATUS: DETECTED (1 Hand: {hand_label}) | Landmarks: {len(h)}/21 | Conf: {h.score:.2f}"
            status_color = (0, 255, 128)  # Bright green
        else:
            status_text = "STATUS: NO HAND DETECTED  [Hand -> null]"
            status_color = (0, 140, 255)  # Orange-amber

        cv2.putText(
            canvas, status_text, (16, 58), self.font, 0.6, status_color, 2, cv2.LINE_AA
        )

        # 2. Real-Time Sign Recognition Indicator Panel (Top-Right)
        if recognitions is not None or dynamic_info is not None:
            self._render_sign_recognition_hud(canvas, recognitions or [], dynamic_info)

        # 3. Coordinate Readout Panel (Bottom-Left)
        if result.hand_detected and result.all_hands:
            if len(result.all_hands) == 1:
                # Single hand panel
                hand = result.all_hands[0]
                wrist = hand[0]
                index_tip = hand[8]
                thumb_tip = hand[4]

                panel_w = 400
                panel_h = 100
                panel_x = 16
                panel_y = height - panel_h - 45

                panel_overlay = canvas.copy()
                cv2.rectangle(
                    panel_overlay,
                    (panel_x, panel_y),
                    (panel_x + panel_w, panel_y + panel_h),
                    (15, 15, 15),
                    -1,
                )
                cv2.addWeighted(panel_overlay, 0.7, canvas, 0.3, 0, canvas)
                cv2.rectangle(
                    canvas,
                    (panel_x, panel_y),
                    (panel_x + panel_w, panel_y + panel_h),
                    (60, 60, 60),
                    1,
                )

                cv2.putText(
                    canvas,
                    f"Coordinates ({hand.handedness or 'Hand'}):",
                    (panel_x + 10, panel_y + 22),
                    self.font,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"L00 (WRIST)     : x={wrist.x:.3f}, y={wrist.y:.3f}, z={wrist.z:.3f}",
                    (panel_x + 10, panel_y + 44),
                    self.font,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"L04 (THUMB_TIP) : x={thumb_tip.x:.3f}, y={thumb_tip.y:.3f}, z={thumb_tip.z:.3f}",
                    (panel_x + 10, panel_y + 64),
                    self.font,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"L08 (INDEX_TIP) : x={index_tip.x:.3f}, y={index_tip.y:.3f}, z={index_tip.z:.3f}",
                    (panel_x + 10, panel_y + 84),
                    self.font,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                # Two hands panel
                panel_w = 460
                panel_h = 120
                panel_x = 16
                panel_y = height - panel_h - 45

                panel_overlay = canvas.copy()
                cv2.rectangle(
                    panel_overlay,
                    (panel_x, panel_y),
                    (panel_x + panel_w, panel_y + panel_h),
                    (15, 15, 15),
                    -1,
                )
                cv2.addWeighted(panel_overlay, 0.7, canvas, 0.3, 0, canvas)
                cv2.rectangle(
                    canvas,
                    (panel_x, panel_y),
                    (panel_x + panel_w, panel_y + panel_h),
                    (60, 60, 60),
                    1,
                )

                cv2.putText(
                    canvas,
                    "Dual-Hand Coordinates (Wrist & Index):",
                    (panel_x + 10, panel_y + 20),
                    self.font,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                for idx, h in enumerate(result.all_hands[:2]):
                    side = h.handedness or f"Hand{idx+1}"
                    w_lm = h[0]
                    i_lm = h[8]
                    y_offset = panel_y + 42 + (idx * 44)
                    cv2.putText(
                        canvas,
                        f"{side:<5} WRIST: ({w_lm.x:.2f}, {w_lm.y:.2f}, {w_lm.z:.2f})",
                        (panel_x + 10, y_offset),
                        self.font,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        f"{side:<5} INDEX: ({i_lm.x:.2f}, {i_lm.y:.2f}, {i_lm.z:.2f})",
                        (panel_x + 10, y_offset + 18),
                        self.font,
                        0.42,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # 4. Bottom Hotkeys Help Bar
        bar_h = 35
        bot_overlay = canvas.copy()
        cv2.rectangle(bot_overlay, (0, height - bar_h), (width, height), (10, 10, 10), -1)
        cv2.addWeighted(bot_overlay, 0.75, canvas, 0.25, 0, canvas)

        help_text = "[Q/Esc] Quit  |  [D] Landmarks  |  [H] HUD  |  [P] Print  |  [S] Save"
        cv2.putText(
            canvas,
            help_text,
            (16, height - 12),
            self.font,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def _render_sign_recognition_hud(
        self,
        canvas: np.ndarray,
        recognitions: list,
        dynamic_info: dict | None = None,
    ) -> None:
        """Render the real-time sign classification card and confidence meter on the top-right."""
        height, width, _ = canvas.shape
        card_w = 260
        card_x = max(16, width - card_w - 16)
        card_y = 82

        if not recognitions and not (dynamic_info and dynamic_info.get("dynamic_prediction")):
            card_h = 74
            overlay = canvas.copy()
            cv2.rectangle(
                overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (18, 18, 18), -1
            )
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
            cv2.rectangle(
                canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1
            )
            cv2.putText(
                canvas,
                "SIGN RECOGNITION",
                (card_x + 12, card_y + 24),
                self.font,
                0.48,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Waiting for hand...",
                (card_x + 12, card_y + 52),
                self.font,
                0.55,
                (0, 170, 255),
                1,
                cv2.LINE_AA,
            )
            return

        # Render a card for each detected hand's recognition
        for idx, rec in enumerate(recognitions[:2]):
            y0 = card_y + idx * 124
            card_h = 114
            if y0 + card_h > height - 45:
                break

            pred = rec.prediction
            stable = rec.stable_label
            display_sign = stable or pred.label or pred.raw_label or "?"
            conf = float(pred.confidence)

            if stable is not None:
                badge_color = (0, 230, 118)  # Bright green for confirmed stable sign
                state_str = "CONFIRMED"
            elif getattr(rec, "is_moving", False):
                badge_color = (255, 180, 0)  # Blue-cyan for motion-gated
                state_str = "MOVING"
            elif pred.label is not None:
                badge_color = (0, 215, 255)  # Gold/yellow while stabilizing
                state_str = "STABILIZING..."
            else:
                badge_color = (120, 120, 255)  # Soft red/orange when below confidence threshold
                state_str = "LOW CONF"

            overlay = canvas.copy()
            cv2.rectangle(
                overlay, (card_x, y0), (card_x + card_w, y0 + card_h), (18, 18, 18), -1
            )
            cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
            cv2.rectangle(
                canvas, (card_x, y0), (card_x + card_w, y0 + card_h), badge_color, 2
            )

            # Header: Handedness + Status
            header = f"{rec.hand.upper()} HAND | {state_str}"
            cv2.putText(
                canvas,
                header,
                (card_x + 12, y0 + 22),
                self.font,
                0.45,
                badge_color,
                1,
                cv2.LINE_AA,
            )

            # Big Sign Label
            font_scale = 1.45 if len(str(display_sign)) <= 2 else 0.95
            cv2.putText(
                canvas,
                f"SIGN: {display_sign}",
                (card_x + 12, y0 + 68),
                self.font,
                font_scale,
                (255, 255, 255) if stable is None else badge_color,
                3,
                cv2.LINE_AA,
            )

            # Confidence bar + percentage
            bar_x = card_x + 12
            bar_y = y0 + 86
            bar_w = card_w - 82
            bar_h_px = 12
            cv2.rectangle(
                canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (55, 55, 55), -1
            )
            fill_w = max(0, min(bar_w, int(round(bar_w * conf))))
            if fill_w > 0:
                cv2.rectangle(
                    canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), badge_color, -1
                )
            cv2.putText(
                canvas,
                f"{conf * 100:5.1f}%",
                (bar_x + bar_w + 8, bar_y + 11),
                self.font,
                0.44,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        # Optional dynamic gesture banner if unified sequence predictor is active
        if dynamic_info and dynamic_info.get("dynamic_prediction") is not None:
            dyn_pred = dynamic_info["dynamic_prediction"]
            dyn_stable = dynamic_info.get("dynamic_stable")
            dyn_label = dyn_stable or dyn_pred.label or dyn_pred.raw_label
            dyn_conf = float(dyn_pred.confidence)
            y_dyn = card_y + len(recognitions[:2]) * 124
            if y_dyn + 60 <= height - 45:
                dyn_color = (0, 255, 180) if dyn_stable else (255, 200, 0)
                overlay = canvas.copy()
                cv2.rectangle(
                    overlay, (card_x, y_dyn), (card_x + card_w, y_dyn + 56), (18, 18, 18), -1
                )
                cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
                cv2.rectangle(
                    canvas, (card_x, y_dyn), (card_x + card_w, y_dyn + 56), dyn_color, 2
                )
                cv2.putText(
                    canvas,
                    f"MOTION: {dyn_label} ({dyn_conf * 100:.0f}%)",
                    (card_x + 12, y_dyn + 36),
                    self.font,
                    0.6,
                    dyn_color,
                    2,
                    cv2.LINE_AA,
                )
