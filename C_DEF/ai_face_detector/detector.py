from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .face import Box, detect_largest_face, load_rgb_image, read_camera_exif_score, resize_roi
from .features import FEATURE_NAMES, extract_features, feature_vector
from .manual_roi import crop_manual_face
from .visualization import draw_box, frequency_map, residual_heatmap


MODEL_PATH = Path("models") / "physics_detector.joblib"
TORCH_MODEL_PATH = Path("models") / "torch_detector.pt"


@dataclass
class BranchScore:
    name: str
    real_evidence: float
    ai_evidence: float
    note: str


@dataclass
class DetectionResult:
    image_path: str
    ai_probability: float
    real_probability: float
    label: str
    confidence: float
    model_mode: str
    face_found: bool
    face_box: Tuple[int, int, int, int]
    face_note: str
    branch_scores: List[BranchScore]
    features: Dict[str, float]
    evidence: List[str]
    exif: Dict[str, str]
    annotated_image: np.ndarray
    residual_image: np.ndarray
    frequency_image: np.ndarray

    def to_report_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("annotated_image", None)
        data.pop("residual_image", None)
        data.pop("frequency_image", None)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_report_dict(), ensure_ascii=False, indent=2)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _ramp(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _window_score(value: float, good_low: float, good_high: float, hard_low: float, hard_high: float) -> float:
    if good_low <= value <= good_high:
        return 1.0
    if value < good_low:
        return _ramp(value, hard_low, good_low)
    return 1.0 - _ramp(value, good_high, hard_high)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_model(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        import joblib

        payload = joblib.load(path)
    except Exception:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload
    return {"pipeline": payload, "feature_names": FEATURE_NAMES}


class AIFaceDetector:
    """Analyze a face image and estimate the probability of AI generation."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.torch_model_path: Optional[Path] = None
        self.model_payload: Optional[dict[str, Any]] = None
        self.torch_model = None
        self.torch_device = "cpu"

        if model_path and self.model_path.suffix.lower() in {".pt", ".pth"}:
            self.torch_model_path = self.model_path
            self.torch_model = self._load_torch_model(self.torch_model_path)
        else:
            self.model_payload = _load_model(self.model_path)
            if self.model_payload is None and not model_path and TORCH_MODEL_PATH.exists():
                self.torch_model_path = TORCH_MODEL_PATH
                self.torch_model = self._load_torch_model(TORCH_MODEL_PATH)

    @property
    def model_loaded(self) -> bool:
        return self.model_payload is not None or self.torch_model is not None

    def analyze(self, image_path: str | Path, manual_box: Box | None = None) -> DetectionResult:
        path = Path(image_path)
        rgb = load_rgb_image(path)
        exif_score, exif_data = read_camera_exif_score(path)
        face = crop_manual_face(rgb, manual_box) if manual_box is not None else None
        if face is None:
            face = detect_largest_face(rgb)
        roi = resize_roi(face.image, 256)
        features = extract_features(roi, exif_camera_score=exif_score)

        branches = self._branch_scores(features)
        heuristic_probability = self._heuristic_ai_probability(features, branches)
        evidence = self._evidence_text(features, branches, exif_data, face.found, face.note)
        if manual_box is not None and face.note == "使用手动框选的人脸区域":
            evidence.insert(0, "已启用手动人脸框，后续特征均来自用户划定区域。")

        model_mode = "physics-heuristic"
        ai_probability = heuristic_probability
        if self.model_payload is not None:
            ai_probability = self._predict_with_model(features)
            model_mode = f"trained-model:{self.model_path}"
            evidence.insert(0, "已加载训练权重，最终概率来自训练模型；分支证据用于解释。")
        elif self.torch_model is not None:
            ai_probability = self._predict_with_torch(roi)
            model_mode = f"torch-three-branch:{self.torch_model_path}"
            evidence.insert(0, "已加载 PyTorch 三分支网络权重，最终概率来自深度模型；分支证据用于解释。")
        else:
            evidence.insert(0, "未发现训练权重，当前结果来自物理启发式初始模型。")

        ai_probability, rule_messages = self._apply_experiment_rules(ai_probability, branches, exif_data)
        for message in reversed(rule_messages):
            evidence.insert(1, message)

        confidence_message = self._low_confidence_message(ai_probability)
        if confidence_message:
            evidence.insert(1, confidence_message)

        label = "疑似 AI 生成" if ai_probability >= 0.5 else "疑似真实拍摄"
        confidence = abs(ai_probability - 0.5) * 2.0
        annotated = draw_box(rgb, face.box, face.found, f"AI {ai_probability * 100:.1f}%")
        return DetectionResult(
            image_path=str(path),
            ai_probability=float(ai_probability),
            real_probability=float(1.0 - ai_probability),
            label=label,
            confidence=float(confidence),
            model_mode=model_mode,
            face_found=face.found,
            face_box=face.box,
            face_note=face.note,
            branch_scores=branches,
            features=features,
            evidence=evidence,
            exif=exif_data,
            annotated_image=annotated,
            residual_image=residual_heatmap(roi),
            frequency_image=frequency_map(roi),
        )

    def _load_torch_model(self, path: Path):
        from .torch_net import load_torch_model

        return load_torch_model(path, device=self.torch_device)

    def _predict_with_model(self, features: dict[str, float]) -> float:
        assert self.model_payload is not None
        pipeline = self.model_payload["pipeline"]
        names = self.model_payload.get("feature_names", FEATURE_NAMES)
        vector = np.asarray([feature_vector(features, names)], dtype=np.float32)
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(vector)[0]
            classes = list(getattr(pipeline, "classes_", [0, 1]))
            if 1 in classes:
                return _clamp(float(probabilities[classes.index(1)]))
            return _clamp(float(probabilities[-1]))
        score = float(pipeline.predict(vector)[0])
        return _clamp(score)

    def _apply_experiment_rules(
        self,
        ai_probability: float,
        branches: list[BranchScore],
        exif: dict[str, str],
    ) -> tuple[float, list[str]]:
        """Apply experiment-derived warning/prompt rules on top of model output."""
        messages: list[str] = []
        rgb_real = 0.0
        smoothness_ai = 0.0
        for branch in branches:
            if branch.name == "RGB 语义分支" or "RGB" in branch.name.upper():
                rgb_real = max(rgb_real, branch.real_evidence)
            if branch.name == "人脸光滑度分支":
                smoothness_ai = max(smoothness_ai, branch.ai_evidence)
        rgb_saturated = rgb_real >= 0.995
        very_smooth = smoothness_ai >= 0.70
        exif_present = bool(exif)

        if rgb_saturated:
            messages.append(
                f"警告：RGB 分支真实证据达到或接近 100%（当前 {rgb_real * 100:.1f}%）。"
                "根据实验现象，这可能是 AI 图像经过对抗优化后的伪装信号，大概率为 AI 生成。"
            )
            ai_probability = _clamp(ai_probability + 0.3)

        if exif_present:
            fields = "、".join(exif.keys())
            messages.append(f"提示：检测到 EXIF 信息（{fields}），根据实验规则，大概率为真实拍摄。")
            ai_probability = _clamp(ai_probability - 0.3)

        if very_smooth:
            messages.append(
                f"警告：人脸光滑度异常偏高（AI 风险 {smoothness_ai * 100:.1f}%）。"
                "可能存在 AI 生成、美颜磨皮、重度降噪或过度平滑处理，已提高 AI 概率。"
            )
            ai_probability = _clamp(ai_probability + 0.75 * _ramp(smoothness_ai, 0.70, 0.99))

        if rgb_saturated and exif_present:
            messages.append("复核提示：该图同时触发 RGB 满分风险与 EXIF 真实提示，建议结合原始文件来源、物理噪声分支和频域分支复核。")

        return ai_probability, messages

    def _low_confidence_message(self, ai_probability: float) -> str | None:
        """Warn on low-confidence decisions without flattening the model score."""
        confidence = abs(ai_probability - 0.5) * 2.0
        if confidence >= 0.5:
            return None

        return (
            f"提示：算法判定置信度低于 50%（当前 {confidence * 100:.1f}%）。"
            "系统保留算法计算出的 AI 概率与结论，不再使用固定概率校准。"
        )

    def _predict_with_torch(self, roi: np.ndarray) -> float:
        import torch

        from .torch_net import make_torch_inputs

        assert self.torch_model is not None
        inputs = make_torch_inputs(roi)
        rgb = torch.from_numpy(inputs["rgb"]).unsqueeze(0).to(self.torch_device)
        residual = torch.from_numpy(inputs["residual"]).unsqueeze(0).to(self.torch_device)
        frequency = torch.from_numpy(inputs["frequency"]).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            logits = self.torch_model(rgb, residual, frequency)
            return _clamp(float(torch.sigmoid(logits).item()))

    def _smoothness_ai_score(self, f: dict[str, float]) -> float:
        """Estimate suspicious over-smoothness in a face ROI."""
        low_laplacian = 1.0 - _ramp(float(np.log1p(f["laplacian_var"])), float(np.log1p(80.0)), float(np.log1p(1600.0)))
        return _clamp(
            _mean(
                [
                    _ramp(f["smooth_area_ratio"], 0.42, 0.90),
                    1.0 - _ramp(f["texture_energy"], 0.018, 0.16),
                    1.0 - _ramp(f["edge_density"], 0.025, 0.18),
                    1.0 - _ramp(f["residual_abs_mean"], 1.15, 6.0),
                    low_laplacian,
                ]
            )
        )

    def _branch_scores(self, f: dict[str, float]) -> list[BranchScore]:
        physical = _mean(
            [
                _window_score(f["residual_std"], 1.2, 11.0, 0.15, 25.0),
                _window_score(f["residual_entropy"], 2.0, 5.4, 0.4, 6.0),
                _ramp(f["noise_illum_corr"], -0.06, 0.36),
                _window_score(f["cfa_periodicity"], 0.018, 0.22, 0.0, 0.5),
                1.0 - _ramp(f["channel_noise_corr"], 0.35, 0.88),
            ]
        )
        frequency = _mean(
            [
                _window_score(f["fft_high_ratio"], 0.025, 0.34, 0.001, 0.62),
                _window_score(f["spectral_slope"], -3.8, -0.55, -8.0, 1.0),
                1.0 - _ramp(f["directional_anisotropy"], 0.18, 0.82),
                1.0 - _ramp(f["jpeg_blockiness"], 0.42, 2.0),
            ]
        )
        rgb = _mean(
            [
                _window_score(f["laplacian_var"], 35.0, 2600.0, 2.0, 8500.0),
                _window_score(f["edge_density"], 0.025, 0.24, 0.0, 0.48),
                _window_score(f["saturation_mean"], 0.05, 0.62, 0.0, 0.95),
                _window_score(f["texture_energy"], 0.012, 0.42, 0.0, 1.2),
                1.0 - _ramp(f["smooth_area_ratio"], 0.38, 0.92),
            ]
        )
        smoothness_ai = self._smoothness_ai_score(f)
        smoothness_real = 1.0 - smoothness_ai

        physical = _clamp(0.88 * physical + 0.12 * f["exif_camera_score"])
        return [
            BranchScore("物理噪声分支", physical, 1.0 - physical, "传感器噪声、CFA 周期、局部噪声-亮度关系"),
            BranchScore("频域残差分支", frequency, 1.0 - frequency, "FFT 高频比例、谱斜率、方向性、JPEG 块效应"),
            BranchScore("RGB 语义分支", rgb, 1.0 - rgb, "清晰度、边缘密度、饱和度、纹理能量"),
            BranchScore("人脸光滑度分支", smoothness_real, smoothness_ai, "局部平滑比例、低纹理能量、低边缘密度、弱残差"),
        ]

    def _heuristic_ai_probability(self, f: dict[str, float], branches: list[BranchScore]) -> float:
        branch_map = {branch.name: branch.real_evidence for branch in branches}
        real_score = (
            0.46 * branch_map["物理噪声分支"]
            + 0.32 * branch_map["频域残差分支"]
            + 0.10 * branch_map["RGB 语义分支"]
            + 0.12 * branch_map["人脸光滑度分支"]
        )
        artifact_boost = 0.0
        artifact_boost += 0.10 * (1.0 - _window_score(f["residual_std"], 1.0, 13.0, 0.0, 32.0))
        artifact_boost += 0.08 * _ramp(f["channel_noise_corr"], 0.58, 0.96)
        artifact_boost += 0.08 * _ramp(f["smooth_area_ratio"], 0.55, 0.96)
        artifact_boost += 0.06 * _ramp(f["directional_anisotropy"], 0.42, 0.95)
        artifact_boost -= 0.10 * f["exif_camera_score"]

        ai_probability = 1.0 - real_score + artifact_boost
        return _clamp(ai_probability, 0.02, 0.98)

    def _evidence_text(
        self,
        f: dict[str, float],
        branches: list[BranchScore],
        exif: dict[str, str],
        face_found: bool,
        face_note: str,
    ) -> list[str]:
        evidence = []
        if face_note == "使用手动框选的人脸区域":
            evidence.append("人脸区域来自手动框选，未使用自动人脸定位结果。")
        else:
            evidence.append("人脸定位成功，分析区域为最大人脸 ROI。" if face_found else "未定位到标准正脸，已改用中心区域，结果应谨慎参考。")
        for branch in branches:
            evidence.append(f"{branch.name}: 真实成像证据 {branch.real_evidence * 100:.1f}%，{branch.note}。")
        if exif:
            keys = "、".join(exif.keys())
            evidence.append(f"图像包含 EXIF 元数据字段：{keys}。")
        else:
            evidence.append("未读取到相机 EXIF 元数据；这不是 AI 证据，但会降低真实设备来源的辅助证据。")

        if f["residual_std"] < 1.0:
            evidence.append("噪声残差很弱，可能存在过度平滑、重采样或生成图常见的缺少传感器噪声现象。")
        if f["channel_noise_corr"] > 0.65:
            evidence.append("RGB 通道残差相关性偏高，独立传感器噪声特征不足。")
        if f["fft_high_ratio"] < 0.02:
            evidence.append("高频能量偏低，细节可能被平滑或生成过程抹除。")
        if f["jpeg_blockiness"] > 0.7:
            evidence.append("JPEG 块效应较明显，压缩可能影响检测稳定性。")
        return evidence
