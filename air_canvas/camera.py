"""Threaded webcam capture for stable UI performance."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Continuously captures webcam frames on a background thread."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "CameraStream":
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self) -> None:
        while self.running:
            success, frame = self.capture.read()
            if success:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()

