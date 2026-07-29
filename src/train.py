"""Train every algorithm x imbalance-strategy combination, score each
on the validation set, and save results + models for the dashboard."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import MODEL_NAMES, SUPPORTS_CLASS_WEIGHT, build_model
from src.resampling import resample

STRATEGIES = ["baseline", "class_weight", "undersample", "oversample", "smote", "adasyn", "smotetomek"]

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "hour_of_day"]
TARGET_COL = "Class"

MODELS_DIR = Path("models")
RESULTS_PATH = Path("results/model_comparison.csv")


def get_scores(model, X):
    """Probability/decision scores used for ROC-AUC and PR-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(y_true, y_pred, y_scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc": average_precision_score(y_true, y_scores),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def run_all(train_df, val_df):
    X_train_full = train_df[FEATURE_COLS]
    y_train_full = train_df[TARGET_COL]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    rows = []
    for model_name in MODEL_NAMES:
        for strategy in STRATEGIES:
            if strategy == "class_weight" and model_name not in SUPPORTS_CLASS_WEIGHT:
                print(f"skip {model_name}/{strategy}: no class_weight support")
                continue

            run_name = f"{model_name}__{strategy}"
            print(f"training {run_name} ...")

            X_res, y_res = resample(X_train_full, y_train_full, strategy)
            model = build_model(model_name, use_class_weight=(strategy == "class_weight"))
            model.fit(X_res, y_res)

            y_scores = get_scores(model, X_val)
            y_pred = model.predict(X_val)
            metrics = evaluate(y_val, y_pred, y_scores)

            joblib.dump(model, MODELS_DIR / f"{run_name}.joblib")
            rows.append({"model": model_name, "strategy": strategy, **metrics})

    results_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(results_df[["model", "strategy", "pr_auc", "roc_auc", "f1"]].to_string(index=False))
    return results_df


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    run_all(train_df, val_df)
