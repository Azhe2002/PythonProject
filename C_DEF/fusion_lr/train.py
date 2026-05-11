"""用提取的四分支分数训练逻辑回归融合权重，与手工权重对比。

输入: models/{dataset}_train_scores.npy + labels.npy（由 extract_branch_scores.py 生成）
输出: models/{dataset}_weights.json，控制台打印对比表格
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(Path(__file__).with_name("config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_data(dataset_name: str, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    models_dir = ROOT / "fusion_lr" / "models"
    score_path = models_dir / f"{dataset_name}_{split}_scores.npy"
    label_path = models_dir / f"{dataset_name}_{split}_labels.npy"
    if score_path.exists() and label_path.exists():
        return np.load(score_path), np.load(label_path)
    return None


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
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


def manual_predict(scores: np.ndarray, manual_weights: dict) -> np.ndarray:
    """用 METHODOLOGY.md 中的手工权重计算 AI 概率。

    real_score = w1*P + w2*F + w3*R + w4*S
    ai_prob = 1 - real_score (忽略 artifact_boost，纯四分支比较)
    """
    w = np.array([
        manual_weights["physical"],
        manual_weights["frequency"],
        manual_weights["rgb"],
        manual_weights["smoothness"],
    ])
    real_score = scores @ w
    return np.clip(1.0 - real_score, 0.02, 0.98)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ece": ece_score(y_true, y_prob),
    }


def print_row(name: str, metrics: dict, weights: list[float] | None = None):
    row = f"  {name:<20} | {metrics['auc']:.4f} | {metrics['f1']:.4f} | {metrics['accuracy']:.4f} | {metrics['ece']:.4f}"
    if weights is not None:
        row += f" | {'/'.join(f'{w:.3f}' for w in weights)}"
    print(row)


def train_single(scores: np.ndarray, labels: np.ndarray, C: float, use_poly: bool, degree: int, random_state: int) -> Pipeline:
    steps: list = []
    if use_poly:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=5000, random_state=random_state)))
    pipe = Pipeline(steps)
    pipe.fit(scores, labels)
    return pipe


def get_weights(pipe: Pipeline, n_branches: int = 4) -> np.ndarray:
    """从 pipeline 中提取四个分支的权重（已包含多项式变换后的首层权重）。"""
    lr = pipe.named_steps["lr"]
    coef = lr.coef_[0]
    scaler = pipe.named_steps["scaler"]
    if "poly" in pipe.named_steps:
        # 取一阶项（前 n_branches 个），忽略交互项
        coef_raw = coef[:n_branches]
    else:
        coef_raw = coef
    # 还原标准化：w_original = coef / scale
    return coef_raw / scaler.scale_


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """相对重要性：|w_i| / sum(|w|)，始终非负且和为 1。

    这不保留符号——符号信息在原始系数中查看。
    用于衡量每个分支对模型决策的相对贡献大小。
    """
    total = np.sum(np.abs(w))
    if total < 1e-10:
        return np.zeros_like(w)
    return np.abs(w) / total


def run_experiment(dataset_name: str, config: dict) -> dict | None:
    manual_weights = config["manual_weights"]
    train_cfg = config["training"]
    C_values = train_cfg["C_values"]
    cv_folds = train_cfg["cv_folds"]
    random_state = train_cfg["random_state"]
    use_poly = train_cfg.get("use_polynomial_features", False)
    poly_degree = train_cfg.get("poly_degree", 2)

    train_data = load_data(dataset_name, "train")
    if train_data is None:
        print(f"  跳过 {dataset_name}: 缺少 train 数据，请先运行 extract_branch_scores.py")
        return None

    X_train, y_train = train_data
    valid_data = load_data(dataset_name, "valid")
    test_data = load_data(dataset_name, "test")

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"  训练集: {X_train.shape[0]} 张 (real={int(np.sum(y_train == 0))}, fake={int(np.sum(y_train == 1))})")
    if valid_data is not None:
        print(f"  验证集: {valid_data[0].shape[0]} 张")
    if test_data is not None:
        print(f"  测试集: {test_data[0].shape[0]} 张")

    # ── 手工权重基线 ──
    print(f"\n  {'指标':<20} | {'AUC':>6} | {'F1':>6} | {'Acc':>6} | {'ECE':>6} | 相对重要性 P/F/R/S (和=1)")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*32}")

    if test_data is not None:
        X_eval, y_eval = test_data
    else:
        X_eval, y_eval = X_train, y_train

    manual_prob = manual_predict(X_eval, manual_weights)
    manual_metrics = evaluate(y_eval, manual_prob)
    print_row("手工权重基线", manual_metrics, [
        manual_weights["physical"],
        manual_weights["frequency"],
        manual_weights["rgb"],
        manual_weights["smoothness"],
    ])

    # ── 交叉验证选最优 C ──
    best_c = C_values[0]
    best_auc = 0.0
    print(f"\n  交叉验证 (C={C_values}, folds={cv_folds}):")

    for C in C_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=5000, random_state=random_state)),
        ])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_auc = float(np.mean(cv_results["test_score"]))
        print(f"    C={C:8.4f}  →  mean AUC={mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_c = C

    print(f"  最优 C = {best_c} (mean AUC = {best_auc:.4f})")

    # ── 最终模型 ──
    pipe = train_single(X_train, y_train, best_c, use_poly, poly_degree, random_state)
    lr_prob = pipe.predict_proba(X_eval)[:, 1]
    lr_metrics = evaluate(y_eval, lr_prob)
    learned_w = get_weights(pipe)
    importance = normalize_weights(learned_w)
    print_row("逻辑回归融合", lr_metrics, list(importance))

    # ── 权重对比 ──
    manual_w = np.array([manual_weights["physical"], manual_weights["frequency"],
                         manual_weights["rgb"], manual_weights["smoothness"]])
    print(f"\n  权重对比:")
    print(f"    手工权重 (和=1): P={manual_w[0]:.4f}  F={manual_w[1]:.4f}  R={manual_w[2]:.4f}  S={manual_w[3]:.4f}")
    print(f"    LR 原始系数:       P={learned_w[0]:.4f}  F={learned_w[1]:.4f}  R={learned_w[2]:.4f}  S={learned_w[3]:.4f}")
    print(f"    LR 相对重要性:     P={importance[0]:.4f}  F={importance[1]:.4f}  R={importance[2]:.4f}  S={importance[3]:.4f}  (和=1)")
    print(f"    截距: {pipe.named_steps['lr'].intercept_[0]:.4f}")

    # ── 保存权重 ──
    models_dir = ROOT / "fusion_lr" / "models"
    result = {
        "dataset": dataset_name,
        "C": best_c,
        "weights_raw": learned_w.tolist(),
        "importance": importance.tolist(),
        "intercept": float(pipe.named_steps["lr"].intercept_[0]),
        "manual_weights": manual_w.tolist(),
        "metrics_lr": lr_metrics,
        "metrics_manual": manual_metrics,
        "branch_names": ["physical", "frequency", "rgb", "smoothness"],
    }
    weight_path = models_dir / f"lr_weights_{dataset_name}.json"
    with open(weight_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"  权重已保存至 {weight_path}")

    return result


def main():
    config = load_config()
    results = {}
    for dataset_name in config["datasets"]:
        result = run_experiment(dataset_name, config)
        if result:
            results[dataset_name] = result

    # ── 汇总 ──
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("跨数据集权重稳定性")
        print(f"{'='*60}")
        all_w = np.array([r["importance"] for r in results.values()])
        names = results[next(iter(results))]["branch_names"]
        print(f"  {'分支':<12} | {'均值':>8} | {'标准差':>8}")
        for i, name in enumerate(names):
            print(f"  {name:<12} | {all_w[:, i].mean():8.4f} | {all_w[:, i].std():8.4f}")


if __name__ == "__main__":
    main()
