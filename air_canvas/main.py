"""Main application entry point for AirCanvas Pro.

This file intentionally stays small at the edges and expressive in the middle:
camera frames come in, hand gestures are interpreted, the canvas is updated,
and the UI is drawn back on top. The lower-level details live in the nearby
modules so the main loop reads like the app's story.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from air_canvas import config
from air_canvas.camera import CameraStream
from air_canvas.canvas_manager import CanvasManager
from air_canvas.gesture_detector import GestureDetector, GestureMode
from air_canvas.hand_tracker import HandData, HandTracker, Point
from air_canvas.toolbar import Toolbar, ToolbarAction


class AirCanvasApp:
    """Coordinates the camera, hand tracking, drawing tools, UI, and shortcuts."""

    def __init__(self) -> None:
        self.camera = CameraStream(
            config.CAMERA_INDEX,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.FPS_TARGET,
        )
        self.tracker = HandTracker(
            config.MAX_NUM_HANDS,
            config.MIN_DETECTION_CONFIDENCE,
            config.MIN_TRACKING_CONFIDENCE,
            config.SMOOTHING_FACTOR,
        )
        self.gestures = GestureDetector()
        self.canvas = CanvasManager(
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.HISTORY_LIMIT,
        )
        self.toolbar = Toolbar(config.TOOLBAR_HEIGHT, config.COLORS, config.BRUSH_SIZES)
        self.selected_color_name = "Black"
        self.brush_size = config.BRUSH_SIZE
        self.eraser_size = config.ERASER_SIZE
        self.mode = GestureMode.IDLE
        self.last_toolbar_action = 0.0
        self.last_save_path: Optional[str] = None
        self.fps = 0.0
        self.previous_frame_time = time.perf_counter()

    def run(self) -> None:
        self.camera.start()
        cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.APP_NAME, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
                hands, results = self.tracker.process(frame)
                output = self._update(frame, hands, results)

                cv2.imshow(config.APP_NAME, output)
                if self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self.camera.stop()
            self.tracker.close()
            cv2.destroyAllWindows()

    def _update(self, frame, hands: list[HandData], results: object):
        primary_hand = hands[0] if hands else None
        cursor = primary_hand.index_tip if primary_hand else None

        # The first visible hand drives the experience. A second hand can still
        # be tracked by MediaPipe, but keeping one "active" hand makes the UI
        # predictable and easier to use.
        if primary_hand is None:
            self.mode = GestureMode.IDLE
            self.canvas.end_stroke()
        else:
            gesture = self.gestures.detect(primary_hand.fingers)
            self.mode = gesture.mode
            self._apply_gesture(primary_hand)

        output = self.canvas.composite(frame)
        self.tracker.draw_landmarks(output, results)
        self.toolbar.draw(output, self.selected_color_name, self.brush_size, self.eraser_size, cursor)
        self._draw_feedback(output, primary_hand)
        self._update_fps()
        return output

    def _apply_gesture(self, hand: HandData) -> None:
        if self.mode == GestureMode.DRAWING and hand.index_tip[1] > config.TOOLBAR_HEIGHT:
            self.canvas.draw_line(hand.index_tip, config.COLORS[self.selected_color_name], self.brush_size)
            return

        if self.mode == GestureMode.SELECTING:
            self.canvas.end_stroke()
            self._handle_toolbar_cursor(hand.index_tip)
            return

        if self.mode == GestureMode.ERASING:
            self.canvas.erase_at(hand.palm_center, self.eraser_size)
            return

        self.canvas.end_stroke()

    def _handle_toolbar_cursor(self, point: Point) -> None:
        if point[1] > config.TOOLBAR_HEIGHT:
            return

        now = time.perf_counter()
        # A small cooldown prevents one fingertip hover from firing the same
        # button many times per second.
        if now - self.last_toolbar_action < 0.45:
            return

        action = self.toolbar.action_at(point)
        if action is None:
            return

        self._dispatch_action(action)
        self.last_toolbar_action = now

    def _dispatch_action(self, action: ToolbarAction) -> None:
        if action.kind == "color" and isinstance(action.value, str):
            self.selected_color_name = action.value
        elif action.kind == "size" and isinstance(action.value, int):
            self.brush_size = action.value
        elif action.kind == "save":
            self.last_save_path = str(self.canvas.save(config.SAVE_DIR))
        elif action.kind == "undo":
            self.canvas.undo()
        elif action.kind == "redo":
            self.canvas.redo()
        elif action.kind == "clear":
            self.canvas.clear()
        elif action.kind == "eraser":
            self.mode = GestureMode.ERASING

    def _draw_feedback(self, frame, hand: Optional[HandData]) -> None:
        self._draw_status_panel(frame, hand)

        if self.last_save_path:
            self._draw_toast(frame, f"Saved: {self.last_save_path}")

        if hand is None:
            return

        # These previews are visual only. The actual brush and eraser behavior
        # lives in CanvasManager, which keeps the drawing logic isolated.
        if self.mode == GestureMode.DRAWING:
            radius = max(8, self.brush_size // 2)
            cv2.circle(frame, hand.index_tip, radius + 5, (8, 12, 20), 2, cv2.LINE_AA)
            cv2.circle(frame, hand.index_tip, radius, config.COLORS[self.selected_color_name], 2, cv2.LINE_AA)
            cv2.circle(frame, hand.index_tip, 3, (245, 248, 255), -1, cv2.LINE_AA)
        elif self.mode == GestureMode.SELECTING:
            cv2.circle(frame, hand.index_tip, 18, (8, 12, 20), 2, cv2.LINE_AA)
            cv2.circle(frame, hand.index_tip, 12, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (hand.index_tip[0] - 20, hand.index_tip[1]), (hand.index_tip[0] + 20, hand.index_tip[1]), (0, 210, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (hand.index_tip[0], hand.index_tip[1] - 20), (hand.index_tip[0], hand.index_tip[1] + 20), (0, 210, 255), 1, cv2.LINE_AA)
        elif self.mode == GestureMode.ERASING:
            cv2.circle(frame, hand.palm_center, self.eraser_size, (12, 16, 24), 3, cv2.LINE_AA)
            cv2.circle(frame, hand.palm_center, self.eraser_size, (238, 242, 248), 2, cv2.LINE_AA)
            cv2.circle(frame, hand.palm_center, 4, (238, 242, 248), -1, cv2.LINE_AA)

    def _draw_status_panel(self, frame, hand: Optional[HandData]) -> None:
        x, y, width, height = 18, 116, 300, 106
        self._rounded_panel(frame, (x, y, width, height), alpha=0.72)

        mode_color = {
            GestureMode.DRAWING: (0, 210, 255),
            GestureMode.SELECTING: (255, 190, 70),
            GestureMode.ERASING: (245, 245, 245),
            GestureMode.IDLE: (145, 155, 175),
        }[self.mode]

        cv2.circle(frame, (x + 22, y + 28), 7, mode_color, -1, cv2.LINE_AA)
        cv2.putText(frame, self.mode.value, (x + 40, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (244, 247, 252), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {self.fps:.1f}", (x + 20, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 225), 1, cv2.LINE_AA)

        confidence = 0 if hand is None else int(hand.confidence * 100)
        cv2.putText(frame, f"Tracking {confidence}%", (x + 130, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 225), 1, cv2.LINE_AA)
        self._draw_confidence_bar(frame, (x + 20, y + 84), confidence)

    def _draw_confidence_bar(self, frame, origin: tuple[int, int], confidence: int) -> None:
        x, y = origin
        width, height = 260, 8
        cv2.rectangle(frame, (x, y), (x + width, y + height), (46, 52, 70), -1, cv2.LINE_AA)
        filled = int(width * max(0, min(confidence, 100)) / 100)
        color = (0, 210, 255) if confidence >= 65 else (255, 190, 70)
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1, cv2.LINE_AA)

    def _draw_toast(self, frame, message: str) -> None:
        width = min(760, max(360, len(message) * 9))
        x = 18
        y = frame.shape[0] - 66
        self._rounded_panel(frame, (x, y, width, 42), alpha=0.76)
        cv2.putText(frame, message, (x + 18, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 255, 170), 1, cv2.LINE_AA)

    def _rounded_panel(self, frame, rect: tuple[int, int, int, int], alpha: float) -> None:
        x, y, width, height = rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (12, 15, 24), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (72, 82, 108), 1, cv2.LINE_AA)

    def _update_fps(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.previous_frame_time
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps = instant_fps if self.fps == 0 else self.fps * 0.9 + instant_fps * 0.1
        self.previous_frame_time = now

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return True
        if key == ord("c"):
            self.canvas.clear()
        elif key == ord("s"):
            self.last_save_path = str(self.canvas.save(config.SAVE_DIR))
        elif key == ord("u") or key == 26:
            self.canvas.undo()
        elif key == ord("r") or key == 25:
            self.canvas.redo()
        return False


def main() -> None:
    AirCanvasApp().run()


if __name__ == "__main__":
    main()
