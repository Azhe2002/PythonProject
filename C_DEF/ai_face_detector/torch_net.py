from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def make_torch_inputs(rgb: np.ndarray, size: int = 256) -> Dict[str, np.ndarray]:
    """Create RGB, physical residual, and frequency inputs for the CNN."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    rgb_tensor = resized.astype(np.float32) / 255.0
    rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.15)
    residual = np.clip(residual / 24.0, -1.0, 1.0)[None, :, :].astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    frequency = magnitude[None, :, :].astype(np.float32)

    return {"rgb": rgb_tensor, "residual": residual, "frequency": frequency}


def build_model():
    import torch
    from torch import nn

    class ConvBranch(nn.Module):
        def __init__(self, in_channels: int, width: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(width * 2, width * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(width * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.out_dim = width * 4

        def forward(self, x):
            return self.net(x).flatten(1)

    class LightweightMultiBranchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rgb = ConvBranch(3, 16)
            self.residual = ConvBranch(1, 10)
            self.frequency = ConvBranch(1, 10)
            dim = self.rgb.out_dim + self.residual.out_dim + self.frequency.out_dim
            self.head = nn.Sequential(
                nn.Linear(dim, 96),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(96, 1),
            )

        def forward(self, rgb, residual, frequency):
            fused = torch.cat([self.rgb(rgb), self.residual(residual), self.frequency(frequency)], dim=1)
            return self.head(fused).squeeze(1)

    return LightweightMultiBranchNet()


def load_torch_model(path: str | Path, device: str = "cpu"):
    import torch

    checkpoint = torch.load(path, map_location=device)
    model = build_model().to(device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model
