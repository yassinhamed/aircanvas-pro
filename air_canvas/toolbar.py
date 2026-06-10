"""Gesture-friendly top toolbar UI.

The toolbar is drawn with OpenCV primitives so the whole app can stay inside a
single webcam window. Buttons are intentionally large because fingertip input
is less precise than a mouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass
class ToolbarAction:
    """Action emitted by toolbar hit testing."""

    kind: str
    value: str | int | None = None


@dataclass
class Button:
    """Drawable toolbar button."""

    label: str
    rect: tuple[int, int, int, int]
    action: ToolbarAction
    color: Optional[tuple[int, int, int]] = None

    def contains(self, point: Point) -> bool:
        x, y, width, height = self.rect
        return x <= point[0] <= x + width and y <= point[1] <= y + height


class Toolbar:
    """Renders controls and maps fingertip touches to app actions."""

    def __init__(
        self,
        height: int,
        colors: dict[str, tuple[int, int, int]],
        sizes: list[int],
    ) -> None:
        self.height = height
        self.colors = colors
        self.sizes = sizes
        self.buttons: list[Button] = []
        self.hovered: Optional[Button] = None
        self.section_labels: list[tuple[str, Point]] = []

    def layout(self, frame_width: int) -> None:
        if self.buttons:
            return

        x = 18
        self.section_labels.append(("COLORS", (x, 14)))
        for name, color in self.colors.items():
            self.buttons.append(
                Button(name, (x, 30, 44, 44), ToolbarAction("color", name), color)
            )
            x += 54

        x += 16
        self.section_labels.append(("BRUSH", (x, 14)))
        for size in self.sizes:
            self.buttons.append(
                Button(str(size), (x, 30, 44, 44), ToolbarAction("size", size))
            )
            x += 50

        x = min(x + 16, frame_width - 430)
        self.section_labels.append(("ACTIONS", (x, 14)))
        for label, kind in [
            ("Eraser", "eraser"),
            ("Save", "save"),
            ("Undo", "undo"),
            ("Redo", "redo"),
            ("Clear", "clear"),
        ]:
            self.buttons.append(Button(label, (x, 30, 76, 44), ToolbarAction(kind)))
            x += 84

    def draw(
        self,
        frame: np.ndarray,
        selected_color: str,
        selected_size: int,
        eraser_size: int,
        cursor: Optional[Point],
    ) -> None:
        self.layout(frame.shape[1])
        self._draw_background(frame)

        self.hovered = self.hit_test(cursor) if cursor else None
        for label, position in self.section_labels:
            cv2.putText(
                frame,
                label,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (155, 165, 185),
                1,
                cv2.LINE_AA,
            )

        for button in self.buttons:
            self._draw_button(frame, button, selected_color, selected_size)

        cv2.putText(
            frame,
            f"AI Air Canvas Pro  |  Eraser {eraser_size}px",
            (18, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (215, 224, 238),
            1,
            cv2.LINE_AA,
        )

    def _draw_button(
        self,
        frame: np.ndarray,
        button: Button,
        selected_color: str,
        selected_size: int,
    ) -> None:
        x, y, width, height = button.rect
        is_selected = (
            button.action.kind == "color"
            and button.action.value == selected_color
            or button.action.kind == "size"
            and button.action.value == selected_size
        )
        is_hovered = self.hovered is button
        border = (0, 210, 255) if is_selected else (86, 92, 112)
        fill = (58, 68, 90) if is_hovered else (30, 34, 46)
        thickness = 3 if is_selected or is_hovered else 1

        self._rounded_rect(frame, (x + 2, y + 3, width, height), (5, 7, 12), -1, 12)
        self._rounded_rect(frame, button.rect, fill, -1, 12)
        self._rounded_rect(frame, button.rect, border, thickness, 12)

        if is_selected:
            cv2.line(frame, (x + 12, y + height - 5), (x + width - 12, y + height - 5), (0, 210, 255), 2, cv2.LINE_AA)

        if button.color is not None:
            center = (x + width // 2, y + height // 2)
            cv2.circle(frame, center, 17, (12, 14, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 14, button.color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 14, (238, 242, 248), 1, cv2.LINE_AA)
            return

        label = button.label
        scale = 0.5 if len(label) <= 4 else 0.42
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (238, 242, 248), 1, cv2.LINE_AA)

    def _draw_background(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], self.height), (11, 14, 23), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.line(frame, (0, self.height - 1), (frame.shape[1], self.height - 1), (0, 210, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (0, self.height - 4), (frame.shape[1], self.height - 4), (45, 53, 75), 1, cv2.LINE_AA)

    def _rounded_rect(
        self,
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        thickness: int,
        radius: int,
    ) -> None:
        x, y, width, height = rect
        x2 = x + width
        y2 = y + height

        if thickness < 0:
            cv2.rectangle(frame, (x + radius, y), (x2 - radius, y2), color, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x2 - radius, y + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
            return

        cv2.line(frame, (x + radius, y), (x2 - radius, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + radius), (x, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

    def hit_test(self, point: Optional[Point]) -> Optional[Button]:
        if point is None:
            return None
        for button in self.buttons:
            if button.contains(point):
                return button
        return None

    def action_at(self, point: Point) -> Optional[ToolbarAction]:
        button = self.hit_test(point)
        return None if button is None else button.action
