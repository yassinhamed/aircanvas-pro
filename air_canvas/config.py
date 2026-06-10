"""Application-wide configuration for AI Air Canvas Pro."""

from __future__ import annotations

from pathlib import Path


APP_NAME = "AI Air Canvas Pro"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS_TARGET = 30

BRUSH_SIZE = 10
ERASER_SIZE = 50
SMOOTHING_FACTOR = 0.35

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MAX_NUM_HANDS = 2

TOOLBAR_HEIGHT = 96
HISTORY_LIMIT = 25

SAVE_DIR = Path("drawings")

COLORS: dict[str, tuple[int, int, int]] = {
    "Black": (0, 0, 0),
    "Blue": (255, 0, 0),
    "Red": (0, 0, 255),
    "Green": (0, 180, 0),
    "Yellow": (0, 230, 230),
    "Purple": (180, 0, 180),
    "White": (255, 255, 255),
}

BRUSH_SIZES = [2, 5, 10, 20, 30, 50]

