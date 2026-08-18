"""
predict.py
----------
Step 4 - THE FINAL DELIVERABLE.

Give this script ANY git repository (local path or URL - does NOT need to be
one it was trained on) and it will:
  1. Mine + extract features for every file in the repo (extract_features.py,
     cutoff_ratio=1.0, i.e. "use all history up to now" - no future to peek at)
  2. Load the trained model (default: XGBoost, falls back to Random Forest)
  3. Predict bug probability for every file
  4. Print a ranked risk report + explain WHY each top file is risky
  5. Save the full ranked table to a CSV the dashboard can read

Usage:
    python predict.py <repo_path_or_url> [models_dir] [output_csv]

Example:
    python predict.py https://github.com/psf/requests models predictions.csv
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import extract  # reuse Step 2 logic


def load_best_model(models_dir):
    """Prefer XGBoost > Random Forest > Logistic Regression, whichever is on disk."""
    for name, fname, needs_scaler in [
        ("XGBoost", "xgboost.joblib", False),
        ("Random Forest", "random_forest.joblib", False),
        ("Logistic Regression", "logistic_regression.joblib", True),
    ]:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            scaler = None
            if needs_scaler:
                scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
            return name, model, scaler
    raise FileNotFoundError(
        f"No trained model found in '{models_dir}/'. Run train_models.py first."
    )


def explain_row(row, feature_cols, importance, dataset_stats, top_n=5):
    """Rough explainability: rank features by (global importance) x (how far this
    file's value is from the dataset average), so we highlight what's unusual
    AND what the model cares about - not just raw feature importance."""
    scores = []
    for feat in feature_cols:
        imp = importance.get(feat, 0)
        mean = dataset_stats["mean"].get(feat, 0)
        std = dataset_stats["std"].get(feat, 1) or 1
        z = (row[feat] - mean) / std
        contribution = imp * max(z, 0)  # only care about values HIGHER than average
        scores.append((feat, contribution, row[feat]))
    scores.sort(key=lambda t: t[1], reverse=True)
    readable = {
        "loc": "High lines of code",
        "complexity": "High code complexity",
        "function_count": "Many functions in file",
        "avg_function_size": "Large average function size",
        "max_function_size": "Very large function present",
        "dependency_count": "High dependency count",
        "commit_count": "Frequently modified (many commits)",
        "developer_count": "Touched by many developers",
        "lines_added": "Large amount of code added over time",
        "lines_deleted": "Large amount of code removed over time",
        "code_churn": "High code churn",
        "recent_commit_count": "Recently modified often",
        "days_since_last_change": "Long time since last change",
        "previous_bug_count": "Multiple previous bug fixes",
    }
    return [readable.get(f, f) for f, c, v in scores[:top_n] if c > 0]


def bar(pct, width=10):
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def main(repo_path, models_dir="models", output_csv="predictions.csv"):
    tmp_csv = "_predict_tmp_dataset.csv"
    extract(repo_path, tmp_csv, cutoff_ratio=1.0)  # no future window - pure inference

    df = pd.read_csv(tmp_csv)
    feature_cols = json.load(open(os.path.join(models_dir, "feature_columns.json")))
    importance_all = json.load(open(os.path.join(models_dir, "feature_importance.json")))

    model_name, model, scaler = load_best_model(models_dir)
    imp_key = "xgboost" if model_name == "XGBoost" else "random_forest"
    importance = importance_all.get(imp_key, importance_all.get("random_forest", {}))

    X = df[feature_cols].fillna(0)
    X_input = scaler.transform(X) if scaler is not None else X
    df["bug_probability"] = model.predict_proba(X_input)[:, 1]

    dataset_stats = {"mean": X.mean().to_dict(), "std": X.std().to_dict()}
    df["risk_factors"] = df.apply(
        lambda r: explain_row(r, feature_cols, importance, dataset_stats), axis=1
    )

    df = df.sort_values("bug_probability", ascending=False).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 40)
    print("      SOFTWARE BUG PREDICTION")
    print("=" * 40)
    print(f"\nModel used: {model_name}")
    print(f"Files analyzed: {len(df)}\n")
    print("HIGH RISK FILES")
    print("-" * 40)
    for _, row in df.head(10).iterrows():
        pct = row["bug_probability"] * 100
        print(f"\n{row['file']}")
        print(f"Bug Probability: {pct:.0f}%")
        print(f"Risk: {bar(pct)} {pct:.0f}%")
        if row["risk_factors"]:
            print("Main Risk Factors:")
            for rf_ in row["risk_factors"]:
                print(f"  - {rf_}")

    print(f"\nFull ranked table saved to: {output_csv}")
    print("\nNote: this identifies files that deserve review/testing priority.")
    print("It does NOT mean a bug is guaranteed to exist.")

    os.remove(tmp_csv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <repo_path_or_url> [models_dir] [output_csv]")
        sys.exit(1)
    md = sys.argv[2] if len(sys.argv) > 2 else "models"
    oc = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
    main(sys.argv[1], md, oc)
