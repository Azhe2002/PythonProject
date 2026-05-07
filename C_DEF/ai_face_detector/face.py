from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceCrop:
    image: np.ndarray
    box: Box
    found: bool
    note: str


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array and honor EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def read_camera_exif_score(path: str | Path) -> tuple[float, dict[str, str]]:
    """Return weak real-camera evidence from EXIF make/model/lens metadata."""
    tags = {
        271: "Make",
        272: "Model",
        305: "Software",
        306: "DateTime",
        42036: "LensModel",
    }
    data: dict[str, str] = {}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            for key, name in tags.items():
                value = exif.get(key)
                if value:
                    data[name] = str(value)[:120]
    except Exception:
        return 0.0, {}

    camera_fields = [data.get("Make"), data.get("Model"), data.get("LensModel")]
    if any(camera_fields):
        return 1.0, data
    if data:
        return 0.35, data
    return 0.0, data


def detect_largest_face(rgb: np.ndarray, padding: float = 0.28) -> FaceCrop:
    """Detect the largest frontal face; fall back to a centered square crop."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

    faces = ()
    if cascade_path.exists():
        detector = cv2.CascadeClassifier(str(cascade_path))
        if not detector.empty():
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(48, 48),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        pad = int(max(w, h) * padding)
        x0 = max(0, int(x) - pad)
        y0 = max(0, int(y) - pad)
        x1 = min(width, int(x + w) + pad)
        y1 = min(height, int(y + h) + pad)
        crop = rgb[y0:y1, x0:x1]
        return FaceCrop(crop, (x0, y0, x1 - x0, y1 - y0), True, "已检测到最大人脸区域")

    side = int(min(width, height) * 0.82)
    x0 = max(0, (width - side) // 2)
    y0 = max(0, (height - side) // 2)
    crop = rgb[y0 : y0 + side, x0 : x0 + side]
    return FaceCrop(crop, (x0, y0, side, side), False, "未检测到清晰人脸，使用中心区域估计")


def resize_roi(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    interpolation = cv2.INTER_AREA if max(rgb.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interpolation)
