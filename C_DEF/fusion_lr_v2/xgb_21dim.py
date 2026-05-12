"""方法 xgb_21dim: 21维原始特征 → XGBoost 非线性分类。

XGBoost 通过树分裂自然习得条件规则（if-then），
无需手动定义阈值和补偿量。与 LR 方案形成线性/非线性对照。
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV

from _shared import (ROOT, compute_baseline, evaluate, load_config, load_data,
                     load_feature_meta, save_report)


def main():
    config = load_config()
    meta = load_feature_meta()
    train_cfg = config["training"]
    xgb_cfg = train_cfg["xgb_params"]
    random_state = train_cfg["random_state"]
    n_raw = meta.get("n_raw_features", 21)

    param_grid = {
        "max_depth": xgb_cfg["max_depth"],
        "learning_rate": xgb_cfg["learning_rate"],
        "n_estimators": [xgb_cfg["n_estimators"]],
    }

    for dataset_name in config["datasets"]:
        train_data = load_data(dataset_name, "train")
        valid_data = load_data(dataset_name, "valid")
        test_data = load_data(dataset_name, "test")
        if train_data is None:
            print(f"跳过 {dataset_name}: 缺少数据")
            continue

        X_train = train_data[0][:, :n_raw]
        y_train = train_data[1]
        X_test = test_data[0][:, :n_raw] if test_data is not None else X_train
        y_test = test_data[1] if test_data is not None else y_train

        # 用验证集做 GridSearch，不用 CV（XGBoost+CV 太慢）
        if valid_data is not None:
            X_valid = valid_data[0][:, :n_raw]
            y_valid = valid_data[1]
            eval_set = [(X_valid, y_valid)]
        else:
            eval_set = None

        print(f"\n{'='*60}")
        print(f"xgb_21dim — {dataset_name}")
        print(f"  训练集: {X_train.shape[0]}, 验证集: "
              f"{valid_data[0].shape[0] if valid_data else 'N/A'}")
        print(f"  测试集: {X_test.shape[0]}")

        # 基线
        baseline = compute_baseline(dataset_name, config)
        print(f"\n  手工基线: AUC={baseline['auc']:.4f} F1={baseline['f1']:.4f}")

        # GridSearch
        print(f"\n  网格搜索: max_depth={xgb_cfg['max_depth']}, "
              f"lr={xgb_cfg['learning_rate']}")
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            random_state=random_state,
            verbosity=0,
        )

        grid = GridSearchCV(
            xgb, param_grid, cv=3, scoring="roc_auc", verbose=0, n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        print(f"  最优: max_depth={best_params['max_depth']}, "
              f"lr={best_params['learning_rate']}, "
              f"CV AUC={grid.best_score_:.4f}")

        # 最终模型（用最优参数在训练集+验证集上训练）
        if valid_data is not None:
            X_train_final = np.vstack([X_train, X_valid])
            y_train_final = np.hstack([y_train, y_valid])
        else:
            X_train_final, y_train_final = X_train, y_train

        best_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            n_estimators=xgb_cfg["n_estimators"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            random_state=random_state,
            verbosity=0,
        )
        best_xgb.fit(X_train_final, y_train_final)

        prob = best_xgb.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, prob)

        print(f"\n  测试集: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
              f"Acc={metrics['accuracy']:.4f} ECE={metrics['ece']:.4f}")
        print(f"  提升 vs 手工基线: AUC +{metrics['auc'] - baseline['auc']:.4f}")

        # 特征重要性
        col_names = meta.get("columns", [])[:n_raw]
        importance = sorted(zip(col_names, best_xgb.feature_importances_),
                            key=lambda x: -x[1])

        print(f"\n  Top 10 特征重要性:")
        for i, (name, imp) in enumerate(importance[:10], 1):
            print(f"    {i:2d}. {name:<24s} {imp:.4f}")

        save_report("xgb_21dim", dataset_name, metrics, importance,
                     {"max_depth": best_params["max_depth"],
                      "learning_rate": best_params["learning_rate"],
                      "cv_auc": float(grid.best_score_),
                      "baseline_auc": baseline["auc"]})


if __name__ == "__main__":
    main()
