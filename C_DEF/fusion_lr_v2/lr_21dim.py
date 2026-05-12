"""方法 lr_21dim: 21维原始特征直接训练逻辑回归。

跳过4分支聚合，StandardScaler + LogisticRegression(CV选C)。
纯数据驱动，不引入任何物理先验。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _shared import (ROOT, compute_baseline, evaluate, load_config, load_data,
                     load_feature_meta, save_report)


def main():
    config = load_config()
    meta = load_feature_meta()
    train_cfg = config["training"]
    C_values = train_cfg["C_values"]
    cv_folds = train_cfg["cv_folds"]
    random_state = train_cfg["random_state"]
    n_raw = meta.get("n_raw_features", 21)

    for dataset_name in config["datasets"]:
        train_data = load_data(dataset_name, "train")
        test_data = load_data(dataset_name, "test")
        if train_data is None:
            print(f"跳过 {dataset_name}: 缺少数据，请先运行 extract_features.py")
            continue

        X_train_full, y_train = train_data
        X_train = X_train_full[:, :n_raw]  # 只用前21维
        X_test, y_test = test_data if test_data is not None else (X_train_full[:, :n_raw], y_train)
        if test_data is not None:
            X_test = X_test[:, :n_raw]

        print(f"\n{'='*60}")
        print(f"lr_21dim — {dataset_name}")
        print(f"  训练集: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
        print(f"  测试集: {X_test.shape[0]}")

        # 手工基线
        baseline = compute_baseline(dataset_name, config)
        print(f"\n  手工基线 (4分支): AUC={baseline['auc']:.4f} F1={baseline['f1']:.4f}")

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
        col_names = meta.get("columns", [])[:n_raw]
        importance = sorted(zip(col_names, np.abs(coef)), key=lambda x: -x[1])

        print(f"\n  Top 10 特征重要性:")
        for i, (name, imp) in enumerate(importance[:10], 1):
            direction = "+" if coef[col_names.index(name)] > 0 else "-"
            print(f"    {i:2d}. {name:<24s} {imp:.4f} ({direction})")

        save_report("lr_21dim", dataset_name, metrics, importance,
                     {"C": best_c, "cv_auc": best_auc, "n_features": n_raw,
                      "baseline_auc": baseline["auc"]})


if __name__ == "__main__":
    main()
