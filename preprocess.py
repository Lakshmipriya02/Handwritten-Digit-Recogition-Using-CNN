"""Turn a user's canvas drawing into an MNIST-style 28x28 input.

MNIST digits are white-on-black, scaled to fit a 20x20 box, and centred by
centre of mass inside the 28x28 frame. Drawings that skip this step get
misclassified far more often, so we replicate it here.
"""
import numpy as np
from PIL import Image


def _to_gray_on_white(arr: np.ndarray) -> np.ndarray:
    """Return an HxW float array, 0 = black ink, 255 = white paper."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[2] == 4:  # RGBA: composite onto white
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        arr = rgb * alpha + 255.0 * (1.0 - alpha)
    else:
        arr = arr[..., :3].astype(np.float32)
    return arr.mean(axis=2)


def extract_canvas(value):
    """Gradio's Sketchpad may hand back an array or a dict; handle both."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite")
    return value


def to_mnist(canvas, ink_is_dark=True):
    """Return (28x28 float32 array in [0,1], white digit on black) or None if blank."""
    if canvas is None:
        return None
    gray = _to_gray_on_white(canvas)
    ink = (255.0 - gray) if ink_is_dark else gray
    ink = np.clip(ink, 0, 255)
    if ink.max() < 30:  # nothing drawn
        return None

    ys, xs = np.where(ink > 20)
    ink = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    h, w = ink.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(ink.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float32)

    frame = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    frame[top: top + new_h, left: left + new_w] = small

    # Shift so the centre of mass sits at the middle of the frame (like MNIST).
    total = frame.sum()
    cy = (frame.sum(axis=1) * np.arange(28)).sum() / total
    cx = (frame.sum(axis=0) * np.arange(28)).sum() / total
    shift_y, shift_x = int(round(13.5 - cy)), int(round(13.5 - cx))
    frame = np.roll(frame, (shift_y, shift_x), axis=(0, 1))

    return frame / 255.0
