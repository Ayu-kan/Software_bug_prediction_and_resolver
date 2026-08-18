"""
train_models.py
----------------
Step 3 of the pipeline.

Loads final_dataset.csv (produced by extract_features.py) and trains:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost

Notes on why things are done this way (see analysis doc):
  - The chronological split already happened in extract_features.py (features
    come from "the past", the future_bug label comes from "the future"), so
    here we only need a normal held-out test split of the resulting rows.
  - Class imbalance (Problem 7.3) is handled with class_weight="balanced" /
    scale_pos_weight, NOT by only looking at accuracy.
  - We report Precision, Recall, F1, ROC-AUC, PR-AUC and Top-K Recall -
    never accuracy alone.

Usage:
    python train_models.py <final_dataset.csv> [models_output_dir]
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
)

FEATURE_COLS = [
    "loc", "complexity", "function_count", "avg_function_size", "max_function_size",
    "dependency_count", "commit_count", "developer_count", "lines_added",
    "lines_deleted", "code_churn", "recent_commit_count", "days_since_last_change",
    "previous_bug_count",
]
TARGET_COL = "future_bug"


def top_k_recall(y_true, y_scores, k_ratio=0.2):
    """Of the true bugs, what fraction show up in the top k% riskiest files?
    This is the metric that matters most for THIS use case: developers only
    have time to review the top few files, not the whole ranked list."""
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_ratio)))
    order = np.argsort(y_scores)[::-1][:k]
    true_positives_in_top_k = y_true.iloc[order].sum() if hasattr(y_true, "iloc") else y_true[order].sum()
    total_positives = y_true.sum()
    if total_positives == 0:
        return float("nan")
    return true_positives_in_top_k / total_positives


def evaluate(name, y_test, y_pred, y_prob):
    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
    }
    # ROC-AUC / PR-AUC need both classes present in y_test
    if len(set(y_test)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 3)
        metrics["pr_auc"] = round(average_precision_score(y_test, y_prob), 3)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    metrics["top_20pct_recall"] = round(top_k_recall(y_test, y_prob, 0.2), 3)
    return metrics


def main(dataset_csv: str, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(dataset_csv)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"Dataset is missing expected columns: {missing}")
        sys.exit(1)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].astype(int)

    print(f"Dataset: {len(df)} rows | future_bug=1: {y.sum()} ({100*y.mean():.1f}%)")

    min_class_count = int(y.value_counts().min()) if y.nunique() > 1 else 0
    if y.nunique() < 2 or min_class_count < 4:
        print(
            "\nERROR: not enough examples of the minority class to train/evaluate "
            "reliably (need at least ~4 buggy AND ~4 non-buggy files).\n"
            "This is the class-imbalance problem the analysis doc warns about "
            "(section 7.3) - fix it by:\n"
            "  - lowering cutoff_ratio in extract_features.py (e.g. 0.6 instead of 0.8) "
            "so more commits fall in the 'future' labeling window, or\n"
            "  - mining a repo with more history / more bug-fix commits, or\n"
            "  - combining datasets from multiple repos (concatenate their final_dataset.csv files).\n"
        )
        sys.exit(1)

    stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = []

    # ---- 1. Logistic Regression (needs scaled features) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr.predict(X_test_scaled)
    results.append(evaluate("Logistic Regression", y_test, lr_pred, lr_prob))
    joblib.dump(lr, os.path.join(out_dir, "logistic_regression.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    # ---- 2. Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, max_depth=8
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, rf_pred, rf_prob))
    joblib.dump(rf, os.path.join(out_dir, "random_forest.joblib"))

    # ---- 3. XGBoost ----
    xgb_model = None
    try:
        from xgboost import XGBClassifier
        pos = max(1, y_train.sum())
        neg = max(1, len(y_train) - y_train.sum())
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            random_state=42,
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict(X_test)
        results.append(evaluate("XGBoost", y_test, xgb_pred, xgb_prob))
        joblib.dump(xgb_model, os.path.join(out_dir, "xgboost.joblib"))
    except ImportError:
        print("xgboost not installed - skipping (pip install xgboost).")

    # ---- Save comparison + feature importance ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    print("\n=== Model comparison (held-out test set) ===")
    print(results_df.to_string(index=False))

    importance = {}
    importance["random_forest"] = {
        f: float(v) for f, v in zip(FEATURE_COLS, rf.feature_importances_.round(4))
    }
    if xgb_model is not None:
        importance["xgboost"] = {
            f: float(v) for f, v in zip(FEATURE_COLS, xgb_model.feature_importances_.round(4))
        }
    with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    with open(os.path.join(out_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    best = results_df.sort_values("pr_auc", ascending=False).iloc[0]
    print(f"\nBest model by PR-AUC: {best['model']} (pick this for the demo/dashboard)")
    print(f"All models + metrics saved in: {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_models.py <final_dataset.csv> [models_output_dir]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "models"
    main(sys.argv[1], out)
