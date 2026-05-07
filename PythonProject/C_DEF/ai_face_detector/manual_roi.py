from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .face import Box, FaceCrop


@dataclass(frozen=True)
class ManualFaceBox:
    """A user-provided face rectangle in original image coordinates."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_tuple(cls, box: Tuple[int, int, int, int]) -> "ManualFaceBox":
        x, y, w, h = box
        return cls(int(x), int(y), int(w), int(h))

    def as_tuple(self) -> Box:
        return (self.x, self.y, self.w, self.h)


def normalize_box(
    box: Tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    min_size: int = 24,
) -> Optional[ManualFaceBox]:
    """Clamp a manual box to image bounds and reject unusably small boxes."""
    height, width = image_shape[:2]
    x, y, w, h = [int(round(value)) for value in box]

    if w < 0:
        x += w
        w = abs(w)
    if h < 0:
        y += h
        h = abs(h)

    x0 = max(0, min(width - 1, x))
    y0 = max(0, min(height - 1, y))
    x1 = max(0, min(width, x + w))
    y1 = max(0, min(height, y + h))

    if x1 <= x0 or y1 <= y0:
        return None
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return ManualFaceBox(x0, y0, x1 - x0, y1 - y0)


def crop_manual_face(rgb: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[FaceCrop]:
    """Create a FaceCrop from a user-provided rectangle."""
    normalized = normalize_box(box, rgb.shape)
    if normalized is None:
        return None

    x, y, w, h = normalized.as_tuple()
    crop = rgb[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    return FaceCrop(crop, normalized.as_tuple(), True, "使用手动框选的人脸区域")


def parse_box(text: str) -> Box:
    """Parse x,y,w,h text used by the CLI."""
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("manual box must be four comma-separated values: x,y,w,h")
    values = tuple(int(float(part)) for part in parts)
    return values  # type: ignore[return-value]
