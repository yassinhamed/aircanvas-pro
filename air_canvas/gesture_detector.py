"""Gesture classification based on MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GestureMode(str, Enum):
    """High-level interaction modes inferred from fingers."""

    DRAWING = "DRAWING"
    SELECTING = "SELECTING COLOR"
    ERASING = "ERASING"
    IDLE = "IDLE"


@dataclass(frozen=True)
class Gesture:
    """Finger state and resulting mode."""

    fingers: tuple[bool, bool, bool, bool, bool]
    mode: GestureMode


class GestureDetector:
    """Converts finger states into app-level gestures."""

    def detect(self, fingers: tuple[bool, bool, bool, bool, bool]) -> Gesture:
        thumb, index, middle, ring, pinky = fingers

        if index and not thumb and not middle and not ring and not pinky:
            return Gesture(fingers, GestureMode.DRAWING)
        if index and middle and not ring and not pinky:
            return Gesture(fingers, GestureMode.SELECTING)
        if sum(fingers) >= 4:
            return Gesture(fingers, GestureMode.ERASING)
        return Gesture(fingers, GestureMode.IDLE)

