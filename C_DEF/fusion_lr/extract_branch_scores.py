"""从训练图像中提取四分支分数，保存为 .npy 供逻辑回归训练使用。

对每张图像调用现有检测器的四分支评分管道：
  real/ 图像 → 标签 0
  fake/ 图像 → 标签 1

输出: models/{dataset}_scores.npy (N,4) + models/{dataset}_labels.npy (N,)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_face_detector.detector import AIFaceDetector
from ai_face_detector.face import detect_largest_face, load_rgb_image, resize_roi
from ai_face_detector.features import extract_features


def load_config() -> dict:
    with open(Path(__file__).with_name("config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def iter_images(directory: Path):
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def extract_dataset(real_dir: Path, fake_dir: Path, detector: AIFaceDetector) -> tuple[np.ndarray, np.ndarray]:
    scores_all: list[list[float]] = []
    labels_all: list[int] = []
    skipped = 0

    for label, directory in [(0, real_dir), (1, fake_dir)]:
        files = sorted(iter_images(directory))
        total = len(files)
        label_name = ["真实", "伪造"][label]
        print(f"\n  提取 {label_name}图像 ({directory}): {total} 张")

        t0 = time.time()
        for i, img_path in enumerate(files):
            if (i + 1) % 500 == 0 or i == total - 1:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"    [{i + 1}/{total}] {speed:.1f} 张/秒", end="\r" if i < total - 1 else "\n")

            try:
                rgb = load_rgb_image(img_path)
            except Exception:
                skipped += 1
                continue

            face = detect_largest_face(rgb)
            roi = resize_roi(face.image, 256)
            features = extract_features(roi, exif_camera_score=0.0)
            branches = detector._branch_scores(features)
            scores_all.append([b.real_evidence for b in branches])
            labels_all.append(label)

    if skipped:
        print(f"  跳过 {skipped} 张无法读取的图像")
    return np.array(scores_all, dtype=np.float32), np.array(labels_all, dtype=np.int32)


SPLIT_KEYS = [
    ("train", "real_dir", "fake_dir"),
    ("valid", "valid_real_dir", "valid_fake_dir"),
    ("test", "test_real_dir", "test_fake_dir"),
]


def main():
    config = load_config()
    detector = AIFaceDetector()
    models_dir = ROOT / "fusion_lr" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, paths in config["datasets"].items():
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*60}")

        for split_name, key_real, key_fake in SPLIT_KEYS:
            if key_real not in paths or key_fake not in paths:
                continue

            real = ROOT / paths[key_real]
            fake = ROOT / paths[key_fake]

            if not real.is_dir() or not fake.is_dir():
                continue

            print(f"\n  --- {split_name} 集 ---")
            scores, labels = extract_dataset(real, fake, detector)

            stem = f"{dataset_name}_{split_name}"
            score_path = models_dir / f"{stem}_scores.npy"
            label_path = models_dir / f"{stem}_labels.npy"
            np.save(score_path, scores)
            np.save(label_path, labels)

            n_real = int(np.sum(labels == 0))
            n_fake = int(np.sum(labels == 1))
            print(f"  保存: {score_path} ({scores.shape})")
            print(f"  保存: {label_path} ({labels.shape})")
            print(f"  真实: {n_real}  伪造: {n_fake}")

            for i, name in enumerate(["物理噪声", "频域残差", "RGB语义", "人脸光滑度"]):
                real_mean = scores[labels == 0, i].mean()
                fake_mean = scores[labels == 1, i].mean()
                print(f"    {name}: real={real_mean:.4f}  fake={fake_mean:.4f}  diff={real_mean - fake_mean:+.4f}")

    print(f"\n完成。分数已保存至 {models_dir}")


if __name__ == "__main__":
    main()
