"""共享特征提取：21维原始特征 + 4分支分数 + 规则触发器。

输出: models/{dataset}_{split}_features.npy (N, 28)
      models/{dataset}_{split}_labels.npy    (N,)
      models/{dataset}_{split}_meta.json     列名映射

28维构成:
  [0:21]  21维原始特征 (FEATURE_NAMES)
  [21:25] 4分支分数 [P物理, F频域, R语义, S光滑度] (real_evidence)
  [25]    光滑度AI风险分 (smoothness_ai)
  [26]    RGB语义分支 real_evidence (用于RGB饱和检测)
  [27]    EXIF相机分
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
from ai_face_detector.face import detect_largest_face, load_rgb_image, read_camera_exif_score, resize_roi
from ai_face_detector.features import FEATURE_NAMES, extract_features, feature_vector


def load_config() -> dict:
    with open(Path(__file__).with_name("config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def iter_images(directory: Path):
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def extract_split(real_dir: Path, fake_dir: Path,
                  detector: AIFaceDetector) -> tuple[np.ndarray, np.ndarray]:
    features_all: list[list[float]] = []
    labels_all: list[int] = []
    skipped = 0

    for label, directory in [(0, real_dir), (1, fake_dir)]:
        files = sorted(iter_images(directory))
        total = len(files)
        label_name = ["真实", "伪造"][label]
        print(f"\n  提取 {label_name}图像: {total} 张 ({directory})")

        t0 = time.time()
        for i, img_path in enumerate(files):
            if (i + 1) % 500 == 0 or i == total - 1:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"    [{i + 1}/{total}] {speed:.1f} 张/秒",
                      end="\r" if i < total - 1 else "\n")

            try:
                rgb = load_rgb_image(img_path)
            except Exception:
                skipped += 1
                continue

            exif_score, _exif_data = read_camera_exif_score(img_path)
            face = detect_largest_face(rgb)
            roi = resize_roi(face.image, 256)

            f21 = extract_features(roi, exif_camera_score=exif_score)
            raw21 = feature_vector(f21, FEATURE_NAMES)

            branches = detector._branch_scores(f21)
            branch_map = {b.name: b.real_evidence for b in branches}
            p_score = branch_map.get("物理噪声分支", 0.0)
            f_score = branch_map.get("频域残差分支", 0.0)
            r_score = branch_map.get("RGB 语义分支", 0.0)
            s_score = branch_map.get("人脸光滑度分支", 0.0)

            ai_branch_map = {b.name: b.ai_evidence for b in branches}
            smoothness_ai = ai_branch_map.get("人脸光滑度分支", 0.0)

            row = raw21 + [p_score, f_score, r_score, s_score, smoothness_ai, r_score, exif_score]
            features_all.append(row)
            labels_all.append(label)

    if skipped:
        print(f"  跳过 {skipped} 张无法读取的图像")
    return np.array(features_all, dtype=np.float32), np.array(labels_all, dtype=np.int32)


SPLIT_KEYS = [
    ("train", "real_dir", "fake_dir"),
    ("valid", "valid_real_dir", "valid_fake_dir"),
    ("test",  "test_real_dir",  "test_fake_dir"),
]


def main():
    config = load_config()
    detector = AIFaceDetector()
    models_dir = ROOT / "fusion_lr_v2" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 生成列名元数据
    col_names = list(FEATURE_NAMES) + [
        "branch_physical", "branch_frequency", "branch_rgb", "branch_smoothness",
        "smoothness_ai_risk", "rgb_real_score", "exif_camera_raw",
    ]
    with open(models_dir / "feature_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"columns": col_names, "n_raw_features": 21}, fh, ensure_ascii=False, indent=2)

    for dataset_name, paths in config["datasets"].items():
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*60}")

        for split_name, key_real, key_fake in SPLIT_KEYS:
            if key_real not in paths or key_fake not in paths:
                continue
            real_dir = ROOT / paths[key_real]
            fake_dir = ROOT / paths[key_fake]
            if not real_dir.is_dir() or not fake_dir.is_dir():
                continue

            print(f"\n  --- {split_name} 集 ---")
            features, labels = extract_split(real_dir, fake_dir, detector)

            stem = f"{dataset_name}_{split_name}"
            np.save(models_dir / f"{stem}_features.npy", features)
            np.save(models_dir / f"{stem}_labels.npy", labels)

            n_real = int(np.sum(labels == 0))
            n_fake = int(np.sum(labels == 1))
            print(f"  保存: {stem}_features.npy ({features.shape})")
            print(f"  保存: {stem}_labels.npy ({labels.shape})")
            print(f"  真实: {n_real}  伪造: {n_fake}")

    print(f"\n完成。特征已保存至 {models_dir}")


if __name__ == "__main__":
    main()
