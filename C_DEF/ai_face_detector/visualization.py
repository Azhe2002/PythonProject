from __future__ import annotations

import cv2
import numpy as np


def save_rgb(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def draw_box(rgb: np.ndarray, box: tuple[int, int, int, int], found: bool, label: str) -> np.ndarray:
    """Draw face bounding box and label on the image."""
    canvas = rgb.copy()
    x, y, w, h = box
    color = (44, 123, 229)  # blue
    thickness = max(2, int(min(rgb.shape[:2]) * 0.003))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    if label:
        font_scale = max(0.5, min(rgb.shape[:2]) * 0.0012)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness // 2))
        cv2.rectangle(canvas, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(canvas, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, thickness // 2))
    return canvas


def residual_heatmap(roi: np.ndarray) -> np.ndarray:
    """Generate a residual heatmap from the face ROI."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.15)
    residual = gray - blur
    residual_abs = np.abs(residual)
    normalized = cv2.normalize(residual_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def frequency_map(roi: np.ndarray) -> np.ndarray:
    """Generate a frequency-domain visualization from the face ROI."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float64)
    centered = gray - gray.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.log1p(np.abs(spectrum))
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
