"""Precompute SHAP explanations for the best strategy per algorithm, so
the dashboard's per-transaction panel can render instantly instead of
computing SHAP on every click.

Tree/boosting models and the two linear models (logistic regression,
linear SVM) get exact, fast explainers over a large sample. KNN and
Naive Bayes have no closed-form SHAP, so they fall back to
KernelExplainer on a much smaller sample — it's slow because it calls
the model hundreds of times per row.
"""

from pathlib import Path

import joblib
import pandas as pd
import shap

from src.data import build_explain_sample
from src.train import FEATURE_COLS

RESULTS_PATH = Path("results/model_comparison.csv")
MODELS_DIR = Path("models")
SHAP_CACHE_DIR = Path("results/shap_cache")

TREE_MODELS = {"decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"}
LINEAR_MODELS = {"logistic_regression", "svm"}
KERNEL_MODELS = {"knn", "naive_bayes"}

FULL_SAMPLE_LEGIT = 2000
SMALL_SAMPLE_LEGIT = 250  # for the slow KernelExplainer models only


"""The single best-scoring (by PR-AUC) resampling strategy for each algorithm — that's the run 
the dashboard treats as "the" model. Meaning Use the best version of each algorithm for explanations."""
def best_strategy_per_model() -> pd.DataFrame:
    results_df = pd.read_csv(RESULTS_PATH)
    best_idx = results_df.groupby("model")["pr_auc"].idxmax()
    return results_df.loc[best_idx].reset_index(drop=True)



def _positive_class_slice(shap_values, expected_value):
    """Some explainers return one SHAP value per class (shape n x
    features x n_classes, expected_value an array) — sklearn's
    DecisionTree/RandomForest and KernelExplainer both do this, while
    the boosting models and linear models return the fraud class
    directly. Normalize to the fraud (class 1) slice either way."""
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if hasattr(expected_value, "__len__"):
        expected_value = expected_value[1]
    return shap_values, expected_value


def explain_model(model, model_name, X_sample):
    """Returns (shap_values, base_value) for the fraud (positive) class."""
    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return _positive_class_slice(shap_values, explainer.expected_value)

    if model_name in LINEAR_MODELS:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        return _positive_class_slice(shap_values, explainer.expected_value)

    background = shap.sample(X_sample, 50, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    return _positive_class_slice(shap_values, explainer.expected_value)


def run_all():
    val_df = pd.read_csv("data/processed/val.csv")
    full_sample = build_explain_sample(val_df, n_legit=FULL_SAMPLE_LEGIT)
    small_sample = build_explain_sample(val_df, n_legit=SMALL_SAMPLE_LEGIT)

    SHAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    best_runs = best_strategy_per_model()

    for _, run in best_runs.iterrows():
        model_name, strategy = run["model"], run["strategy"]
        run_name = f"{model_name}__{strategy}"
        sample_df = small_sample if model_name in KERNEL_MODELS else full_sample
        print(f"computing SHAP for {run_name} on {len(sample_df)} rows ...")

        model = joblib.load(MODELS_DIR / f"{run_name}.joblib")
        X_sample = sample_df[FEATURE_COLS]
        shap_values, base_value = explain_model(model, model_name, X_sample)

        joblib.dump(
            {
                "run_name": run_name,
                "shap_values": shap_values,
                "base_value": base_value,
                "X_sample": X_sample.reset_index(drop=True),
                "row_index": sample_df.index.to_numpy(),
                "feature_names": FEATURE_COLS,
            },
            SHAP_CACHE_DIR / f"{model_name}.joblib",
        )


if __name__ == "__main__":
    run_all()
