from __future__ import annotations

from typing import Dict, Iterable

import cv2
import numpy as np


FEATURE_NAMES = [
    "residual_std",
    "residual_abs_mean",
    "residual_entropy",
    "residual_kurtosis",
    "noise_illum_corr",
    "cfa_periodicity",
    "channel_noise_corr",
    "fft_low_ratio",
    "fft_mid_ratio",
    "fft_high_ratio",
    "spectral_slope",
    "directional_anisotropy",
    "jpeg_blockiness",
    "laplacian_var",
    "edge_density",
    "saturation_mean",
    "saturation_std",
    "color_channel_corr",
    "smooth_area_ratio",
    "texture_energy",
    "exif_camera_score",
]


def _safe_float(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return 0.0


def _entropy(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values.ravel(), bins=bins, density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    prob = hist.astype(np.float64) / float(total)
    prob = prob[prob > 0]
    return _safe_float(-np.sum(prob * np.log2(prob)))


def _kurtosis(values: np.ndarray) -> float:
    flat = values.astype(np.float64).ravel()
    mean = flat.mean()
    centered = flat - mean
    var = np.mean(centered * centered)
    if var <= 1e-12:
        return 0.0
    return _safe_float(np.mean(centered**4) / (var**2))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    if aa.size < 2 or bb.size < 2:
        return 0.0
    if np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
        return 0.0
    return _safe_float(np.corrcoef(aa, bb)[0, 1])


def _block_stats(gray: np.ndarray, residual: np.ndarray, block: int = 16) -> tuple[float, float]:
    means = []
    stds = []
    h, w = gray.shape
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            means.append(float(gray[y : y + block, x : x + block].mean()))
            stds.append(float(residual[y : y + block, x : x + block].std()))
    if len(means) < 2:
        return 0.0, 0.0
    corr = _corr(np.asarray(means), np.asarray(stds))
    smooth_ratio = float(np.mean(np.asarray(stds) < 1.2))
    return corr, smooth_ratio


def _cfa_periodicity(rgb_float: np.ndarray) -> float:
    scores = []
    for channel in range(3):
        arr = rgb_float[:, :, channel]
        residual = arr - cv2.GaussianBlur(arr, (0, 0), 1.1)
        groups = [
            np.mean(np.abs(residual[0::2, 0::2])),
            np.mean(np.abs(residual[0::2, 1::2])),
            np.mean(np.abs(residual[1::2, 0::2])),
            np.mean(np.abs(residual[1::2, 1::2])),
        ]
        denom = float(np.mean(groups)) + 1e-6
        scores.append(float(np.std(groups) / denom))
    return _safe_float(float(np.mean(scores)))


def _channel_noise_corr(rgb_float: np.ndarray) -> float:
    residuals = []
    for channel in range(3):
        arr = rgb_float[:, :, channel]
        residuals.append(arr - cv2.GaussianBlur(arr, (0, 0), 1.0))
    pairs = [
        abs(_corr(residuals[0], residuals[1])),
        abs(_corr(residuals[0], residuals[2])),
        abs(_corr(residuals[1], residuals[2])),
    ]
    return _safe_float(float(np.mean(pairs)))


def _fft_features(gray: np.ndarray) -> tuple[float, float, float, float, float]:
    centered = gray.astype(np.float64) - float(gray.mean())
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius_norm = radius / (radius.max() + 1e-9)

    total = float(power.sum()) + 1e-9
    low = float(power[radius_norm < 0.16].sum()) / total
    mid = float(power[(radius_norm >= 0.16) & (radius_norm < 0.38)].sum()) / total
    high = float(power[radius_norm >= 0.38].sum()) / total

    bins = np.linspace(0.03, 0.95, 28)
    radial_energy = []
    radial_freq = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (radius_norm >= left) & (radius_norm < right)
        if np.any(mask):
            radial_energy.append(float(power[mask].mean()) + 1e-9)
            radial_freq.append((left + right) / 2.0)
    if len(radial_energy) > 2:
        slope = float(np.polyfit(np.log(radial_freq), np.log(radial_energy), 1)[0])
    else:
        slope = 0.0

    horizontal = power[np.abs(yy - cy) < h * 0.025].sum()
    vertical = power[np.abs(xx - cx) < w * 0.025].sum()
    anisotropy = float(abs(horizontal - vertical) / (horizontal + vertical + 1e-9))
    return tuple(_safe_float(v) for v in (low, mid, high, slope, anisotropy))


def _jpeg_blockiness(gray: np.ndarray) -> float:
    gray = gray.astype(np.float32)
    if min(gray.shape) < 16:
        return 0.0

    h, w = gray.shape
    bx = np.arange(8, w, 8)
    by = np.arange(8, h, 8)
    ix = np.arange(4, w, 8)
    iy = np.arange(4, h, 8)

    vertical_boundary = np.abs(gray[:, bx] - gray[:, bx - 1]).mean() if bx.size else 0.0
    horizontal_boundary = np.abs(gray[by, :] - gray[by - 1, :]).mean() if by.size else 0.0
    vertical_inner = np.abs(gray[:, ix] - gray[:, ix - 1]).mean() if ix.size else 0.0
    horizontal_inner = np.abs(gray[iy, :] - gray[iy - 1, :]).mean() if iy.size else 0.0
    boundary = float((vertical_boundary + horizontal_boundary) * 0.5)
    inner = float((vertical_inner + horizontal_inner) * 0.5) + 1e-6
    return _safe_float(max(0.0, (boundary - inner) / inner))


def _color_channel_corr(rgb_float: np.ndarray) -> float:
    pairs = [
        abs(_corr(rgb_float[:, :, 0], rgb_float[:, :, 1])),
        abs(_corr(rgb_float[:, :, 0], rgb_float[:, :, 2])),
        abs(_corr(rgb_float[:, :, 1], rgb_float[:, :, 2])),
    ]
    return _safe_float(float(np.mean(pairs)))


def extract_features(rgb: np.ndarray, exif_camera_score: float = 0.0) -> Dict[str, float]:
    """Extract RGB, frequency-domain, and physical-noise features from a face ROI."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb_float = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.15)
    residual = gray - blur

    residual_std = float(residual.std())
    residual_abs_mean = float(np.mean(np.abs(residual)))
    residual_entropy = _entropy(np.clip(residual, -32, 32), bins=64)
    residual_kurtosis = _kurtosis(residual)
    noise_illum_corr, smooth_area_ratio = _block_stats(gray, residual)

    fft_low, fft_mid, fft_high, spectral_slope, anisotropy = _fft_features(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian_var = float(laplacian.var())
    edges = cv2.Canny(gray.astype(np.uint8), 80, 160)
    edge_density = float(np.mean(edges > 0))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    texture_energy = float(np.mean(np.abs(laplacian)) / 255.0)

    values = {
        "residual_std": residual_std,
        "residual_abs_mean": residual_abs_mean,
        "residual_entropy": residual_entropy,
        "residual_kurtosis": residual_kurtosis,
        "noise_illum_corr": noise_illum_corr,
        "cfa_periodicity": _cfa_periodicity(rgb_float),
        "channel_noise_corr": _channel_noise_corr(rgb_float),
        "fft_low_ratio": fft_low,
        "fft_mid_ratio": fft_mid,
        "fft_high_ratio": fft_high,
        "spectral_slope": spectral_slope,
        "directional_anisotropy": anisotropy,
        "jpeg_blockiness": _jpeg_blockiness(gray),
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "color_channel_corr": _color_channel_corr(rgb_float),
        "smooth_area_ratio": smooth_area_ratio,
        "texture_energy": texture_energy,
        "exif_camera_score": float(exif_camera_score),
    }
    return {name: _safe_float(values.get(name, 0.0)) for name in FEATURE_NAMES}


def feature_vector(features: dict[str, float], names: Iterable[str] = FEATURE_NAMES) -> list[float]:
    return [float(features.get(name, 0.0)) for name in names]
