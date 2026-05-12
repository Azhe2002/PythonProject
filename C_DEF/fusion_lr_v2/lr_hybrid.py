"""方法 lr_hybrid: 21维原始特征 + 物理先验特征 → 逻辑回归。

在21维基础上追加6维物理知识特征：
  - 4分支分数 (P/F/R/S) — 保留原手工管道的领域知识
  - 光滑度AI风险分 — 原规则 "if smooth>=0.70: +0.75*ramp"
  - RGB语义分 — 原规则 "if rgb>=0.995: +0.3"

模型自行学习这些物理特征的权重，不写死阈值和补偿量。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _shared import (ROOT, compute_baseline, evaluate, load_config, load_data,
                     load_feature_meta, save_report)

# 在原21维之外使用的额外特征列索引
EXTRA_COLS = [21, 22, 23, 24, 25, 26]  # P,F,R,S分支 + 光滑度AI + RGB分
EXTRA_NAMES = ["branch_physical", "branch_frequency", "branch_rgb", "branch_smoothness",
               "smoothness_ai_risk", "rgb_real_score"]


def main():
    config = load_config()
    meta = load_feature_meta()
    train_cfg = config["training"]
    C_values = train_cfg["C_values"]
    cv_folds = train_cfg["cv_folds"]
    random_state = train_cfg["random_state"]
    n_raw = meta.get("n_raw_features", 21)
    all_cols = meta.get("columns", [])

    # 构建特征掩码：21维 + 指定额外列
    feature_mask = list(range(n_raw)) + [n_raw + i for i in range(len(EXTRA_COLS))
                                          if n_raw + i < len(all_cols)]
    feature_names = [all_cols[i] for i in feature_mask]

    for dataset_name in config["datasets"]:
        train_data = load_data(dataset_name, "train")
        test_data = load_data(dataset_name, "test")
        if train_data is None:
            print(f"跳过 {dataset_name}: 缺少数据")
            continue

        X_train = train_data[0][:, feature_mask]
        y_train = train_data[1]
        X_test = test_data[0][:, feature_mask] if test_data is not None else X_train
        y_test = test_data[1] if test_data is not None else y_train

        print(f"\n{'='*60}")
        print(f"lr_hybrid — {dataset_name}")
        print(f"  训练集: {X_train.shape[0]}, 特征维度: {X_train.shape[1]} "
              f"(21原始 + {len(EXTRA_COLS)}物理)")
        print(f"  测试集: {X_test.shape[0]}")

        # 基线
        baseline = compute_baseline(dataset_name, config)
        print(f"\n  手工基线 (4分支+规则): AUC={baseline['auc']:.4f} F1={baseline['f1']:.4f}")

        # CV 选 C
        best_c = C_values[0]
        best_auc = 0.0
        print(f"\n  交叉验证选C ({cv_folds}-fold):")
        for C in C_values:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=5000,
                                          random_state=random_state)),
            ])
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            mean_auc = float(np.mean(cv_results["test_score"]))
            marker = " *" if mean_auc > best_auc else ""
            print(f"    C={C:8.4f} → AUC={mean_auc:.4f}{marker}")
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_c = C

        print(f"  最优 C = {best_c}")

        # 最终模型
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_c, solver="lbfgs", max_iter=5000,
                                      random_state=random_state)),
        ])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, prob)

        print(f"\n  测试集: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
              f"Acc={metrics['accuracy']:.4f} ECE={metrics['ece']:.4f}")
        print(f"  提升 vs 手工基线: AUC +{metrics['auc'] - baseline['auc']:.4f}")

        # 特征重要性
        lr = pipe.named_steps["lr"]
        scaler = pipe.named_steps["scaler"]
        coef = lr.coef_[0] / scaler.scale_
        importance = sorted(zip(feature_names, np.abs(coef)), key=lambda x: -x[1])

        print(f"\n  Top 10 特征重要性:")
        for i, (name, imp) in enumerate(importance[:10], 1):
            direction = "+" if coef[feature_names.index(name)] > 0 else "-"
            tag = " [物理]" if name in EXTRA_NAMES else ""
            print(f"    {i:2d}. {name:<24s} {imp:.4f} ({direction}){tag}")

        # 物理特征权重解读
        print(f"\n  物理特征权重:")
        for name in EXTRA_NAMES:
            if name in feature_names:
                idx = feature_names.index(name)
                raw_w = coef[idx]
                print(f"    {name:<20s}: {raw_w:+.4f}")

        save_report("lr_hybrid", dataset_name, metrics, importance,
                     {"C": best_c, "cv_auc": best_auc, "n_features": len(feature_mask),
                      "n_physics_features": len(EXTRA_COLS),
                      "baseline_auc": baseline["auc"]})


if __name__ == "__main__":
    main()
