"""fusion_lr_v2 共享工具：数据加载、指标计算、报告保存。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(Path(__file__).with_name("config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_data(dataset_name: str, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    models_dir = ROOT / "fusion_lr_v2" / "models"
    feat_path = models_dir / f"{dataset_name}_{split}_features.npy"
    label_path = models_dir / f"{dataset_name}_{split}_labels.npy"
    if feat_path.exists() and label_path.exists():
        return np.load(feat_path), np.load(label_path)
    return None


def load_feature_meta() -> dict:
    meta_path = ROOT / "fusion_lr_v2" / "models" / "feature_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ece": ece_score(y_true, y_prob),
    }


def manual_predict(scores_4branch: np.ndarray, manual_weights: dict) -> np.ndarray:
    w = np.array([
        manual_weights["physical"], manual_weights["frequency"],
        manual_weights["rgb"], manual_weights["smoothness"],
    ])
    real_score = scores_4branch @ w
    return np.clip(1.0 - real_score, 0.02, 0.98)


def compute_baseline(dataset_name: str, config: dict) -> dict:
    """手工权重基线：用4分支分+artifact boost规则计算AI概率。"""
    test_data = load_data(dataset_name, "test")
    if test_data is None:
        train_data = load_data(dataset_name, "train")
        if train_data is None:
            return {"auc": 0.0, "f1": 0.0, "accuracy": 0.0, "ece": 0.0}
        X, y = train_data
    else:
        X, y = test_data

    meta = load_feature_meta()
    branch_cols = meta.get("columns", [])
    p_idx = branch_cols.index("branch_physical") if "branch_physical" in branch_cols else 21
    f_idx = branch_cols.index("branch_frequency") if "branch_frequency" in branch_cols else 22
    r_idx = branch_cols.index("branch_rgb") if "branch_rgb" in branch_cols else 23
    s_idx = branch_cols.index("branch_smoothness") if "branch_smoothness" in branch_cols else 24

    mw = config["manual_weights"]
    scores_4 = X[:, [p_idx, f_idx, r_idx, s_idx]]
    prob = manual_predict(scores_4, mw)
    return evaluate(y, prob)


def save_report(method_name: str, dataset_name: str, metrics: dict,
                feature_importance: list[tuple[str, float]] | None = None,
                extra_info: dict | None = None):
    result_dir = ROOT / "fusion_lr_v2" / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {method_name} — {dataset_name} 训练结果",
        "",
        "## 性能指标",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| AUC | {metrics['auc']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| ECE | {metrics['ece']:.4f} |",
    ]

    if extra_info:
        lines.append("")
        lines.append("## 超参数")
        lines.append("")
        for k, v in extra_info.items():
            lines.append(f"- **{k}**: {v}")

    if feature_importance:
        lines.append("")
        lines.append("## 特征重要性 (Top 15)")
        lines.append("")
        lines.append("| 排名 | 特征 | 重要性 |")
        lines.append("|---|---|---|")
        for i, (name, imp) in enumerate(feature_importance[:15], 1):
            lines.append(f"| {i} | {name} | {imp:.4f} |")

    report_path = result_dir / f"{method_name}.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  报告已保存至 {report_path}")
