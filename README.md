# AirCanvas Pro

Draw on your screen by moving your hand in the air.

AirCanvas Pro is a webcam-based drawing app built with **Python**, **OpenCV**, **MediaPipe Hands**, and **NumPy**. It tracks your hand in real time, turns simple gestures into tools, and lets you sketch without touching a mouse, tablet, or screen.

The goal of this project is simple: make computer vision feel playful, useful, and easy to understand.

## Preview

Use your webcam, raise a finger, and start drawing.

- Index finger: draw.
- Index + middle finger: select colors, pen sizes, and toolbar actions.
- Open palm: erase only the area under your hand, like a real whiteboard eraser.

## Features

- Real-time webcam feed at `1280x720`.
- Hand landmark tracking with MediaPipe Hands.
- Air drawing with smooth line interpolation.
- Color picker with Black, Blue, Red, Green, Yellow, Purple, and White.
- Brush size controls: `2`, `5`, `10`, `20`, `30`, `50`.
- Smart eraser that removes only touched pixels.
- Undo, redo, clear canvas, and timestamped PNG export.
- Modern transparent toolbar and status HUD.
- FPS counter and tracking confidence display.
- Threaded camera capture for smoother performance.

## Project Structure

```text
air_canvas/
├── main.py              # App loop and high-level coordination
├── camera.py            # Threaded webcam capture
├── hand_tracker.py      # MediaPipe wrapper and landmark smoothing
├── gesture_detector.py  # Gesture-to-mode classification
├── canvas_manager.py    # Drawing layer, eraser, undo/redo, saving
├── toolbar.py           # On-screen toolbar and gesture hit testing
├── config.py            # App settings
└── assets/
```

## Installation

Python `3.11+` is recommended.

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m air_canvas.main
```

You can also run the compatibility launcher:

```bash
python personal.py
```

## Controls

### Gestures

| Gesture | Action |
| --- | --- |
| Index finger only | Draw |
| Index + middle finger | Select toolbar buttons |
| Open palm | Erase under palm |
| No active gesture | Idle |

### Keyboard

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Quit |
| `C` | Clear canvas |
| `S` | Save PNG |
| `U` / `Ctrl+Z` | Undo |
| `R` / `Ctrl+Y` | Redo |

## Configuration

Most settings live in `air_canvas/config.py`:

- Camera size and FPS target.
- Default brush and eraser size.
- Gesture smoothing.
- Available colors and brush sizes.
- Save folder.

## Notes

- Good lighting improves hand tracking a lot.
- Keep your hand inside the webcam frame.
- If the toolbar feels too sensitive, increase the cooldown in `air_canvas/main.py`.
- Saved drawings are written to `drawings/`.

## Repository Name Ideas

If you want a clean GitHub name, I recommend:

**`aircanvas-pro`**

Other nice options:

- `gesture-canvas`
- `handpaint-ai`
- `air-draw-studio`
- `opencv-air-canvas`

## Description

Suggested GitHub description:

> A professional Python air-drawing app that uses OpenCV and MediaPipe Hands to turn real-time hand gestures into drawing, erasing, color selection, undo/redo, and PNG export.

