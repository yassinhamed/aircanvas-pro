"""Drawing canvas, erasing, history, and saving logic.

The app draws on a separate layer instead of directly modifying the webcam
frame. That keeps the camera image temporary and the artwork persistent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


Point = tuple[int, int]


class CanvasManager:
    """Manages the transparent artwork layer shown over the live camera feed."""

    def __init__(self, width: int, height: int, history_limit: int) -> None:
        self.width = width
        self.height = height
        self.history_limit = history_limit
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.undo_stack: list[tuple[np.ndarray, np.ndarray]] = []
        self.redo_stack: list[tuple[np.ndarray, np.ndarray]] = []
        self.last_point: Optional[Point] = None
        self.stroke_active = False

    def begin_stroke(self) -> None:
        if not self.stroke_active:
            self.push_history()
            self.stroke_active = True

    def end_stroke(self) -> None:
        self.stroke_active = False
        self.last_point = None

    def draw_line(self, point: Point, color: tuple[int, int, int], size: int) -> None:
        self.begin_stroke()
        if self.last_point is None:
            self.last_point = point

        cv2.line(self.layer, self.last_point, point, color, size, cv2.LINE_AA)
        cv2.line(self.mask, self.last_point, point, 255, size, cv2.LINE_AA)
        self.last_point = point

    def erase_at(self, point: Point, radius: int) -> None:
        self.begin_stroke()
        cv2.circle(self.layer, point, radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.mask, point, radius, 0, -1, cv2.LINE_AA)

    def clear(self) -> None:
        self.push_history()
        self.layer.fill(0)
        self.mask.fill(0)
        self.end_stroke()

    def composite(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        visible = self.mask > 0
        output[visible] = self.layer[visible]
        return output

    def push_history(self) -> None:
        """Store a lightweight snapshot before a stroke-changing action."""
        self.undo_stack.append((self.layer.copy(), self.mask.copy()))
        if len(self.undo_stack) > self.history_limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append((self.layer.copy(), self.mask.copy()))
        self.layer, self.mask = self.undo_stack.pop()
        self.end_stroke()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append((self.layer.copy(), self.mask.copy()))
        self.layer, self.mask = self.redo_stack.pop()
        self.end_stroke()

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = directory / f"drawing_{timestamp}.png"
        white_background = np.full_like(self.layer, 255)
        image = white_background.copy()
        visible = self.mask > 0
        image[visible] = self.layer[visible]
        cv2.imwrite(str(path), image)
        return path
