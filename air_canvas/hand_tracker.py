"""MediaPipe Hands wrapper with pixel landmark conversion and smoothing.

MediaPipe gives normalized landmark coordinates. This module converts them
into screen pixels and exposes only the points the rest of the app cares about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


Point = tuple[int, int]


@dataclass
class HandData:
    """Useful hand state extracted from MediaPipe results."""

    landmarks: list[Point]
    fingers: tuple[bool, bool, bool, bool, bool]
    handedness: str
    confidence: float

    @property
    def index_tip(self) -> Point:
        return self.landmarks[8]

    @property
    def middle_tip(self) -> Point:
        return self.landmarks[12]

    @property
    def thumb_tip(self) -> Point:
        return self.landmarks[4]

    @property
    def palm_center(self) -> Point:
        palm_ids = [0, 5, 9, 13, 17]
        x = int(sum(self.landmarks[index][0] for index in palm_ids) / len(palm_ids))
        y = int(sum(self.landmarks[index][1] for index in palm_ids) / len(palm_ids))
        return x, y


class LandmarkSmoother:
    """Applies exponential smoothing so the cursor feels less nervous."""

    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.previous: Optional[list[Point]] = None

    def smooth(self, points: list[Point]) -> list[Point]:
        if self.previous is None or len(self.previous) != len(points):
            self.previous = points
            return points

        smoothed: list[Point] = []
        for old, new in zip(self.previous, points):
            x = int(old[0] * (1 - self.factor) + new[0] * self.factor)
            y = int(old[1] * (1 - self.factor) + new[1] * self.factor)
            smoothed.append((x, y))

        self.previous = smoothed
        return smoothed


class HandTracker:
    """Tracks hands and returns smoothed, application-friendly hand data."""

    def __init__(
        self,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
        smoothing_factor: float,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.smoothers: dict[int, LandmarkSmoother] = {}
        self.smoothing_factor = smoothing_factor

    def process(self, frame: np.ndarray) -> tuple[list[HandData], object]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        tracked: list[HandData] = []

        if not results.multi_hand_landmarks:
            return tracked, results

        handedness_results = results.multi_handedness or []
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            points = [
                (int(landmark.x * width), int(landmark.y * height))
                for landmark in landmarks.landmark
            ]
            smoother = self.smoothers.setdefault(index, LandmarkSmoother(self.smoothing_factor))
            points = smoother.smooth(points)

            label = "Unknown"
            confidence = 0.0
            if index < len(handedness_results):
                classification = handedness_results[index].classification[0]
                label = classification.label
                confidence = classification.score

            tracked.append(
                HandData(
                    landmarks=points,
                    fingers=self._finger_states(points, label),
                    handedness=label,
                    confidence=confidence,
                )
            )

        return tracked, results

    def _finger_states(
        self, points: list[Point], handedness: str
    ) -> tuple[bool, bool, bool, bool, bool]:
        thumb_is_open = points[4][0] > points[3][0]
        if handedness == "Right":
            thumb_is_open = points[4][0] < points[3][0]

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        fingers = [thumb_is_open]
        fingers.extend(points[tip][1] < points[pip][1] for tip, pip in zip(finger_tips, finger_pips))
        return tuple(fingers)  # type: ignore[return-value]

    def draw_landmarks(self, frame: np.ndarray, results: object) -> None:
        if not getattr(results, "multi_hand_landmarks", None):
            return
        for landmarks in results.multi_hand_landmarks:
            self.drawer.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        self.hands.close()
