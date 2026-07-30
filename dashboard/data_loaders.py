"""Small IO helpers the dashboard reads from the artifacts training
already produced — no retraining or recomputation happens here."""

from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")
RESULTS_PATH = Path("results/model_comparison.csv")
SHAP_CACHE_DIR = Path("results/shap_cache")
EMBEDDING_CACHE_DIR = Path("results/embedding_cache")
PROCESSED_DIR = Path("data/processed")


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["run"] = df["model"] + "__" + df["strategy"]
    return df


def load_model(run_name: str):
    return joblib.load(MODELS_DIR / f"{run_name}.joblib")


def load_shap_cache(model_name: str) -> dict:
    return joblib.load(SHAP_CACHE_DIR / f"{model_name}.joblib")


def load_embedding_cache(method: str) -> dict:
    return joblib.load(EMBEDDING_CACHE_DIR / f"{method}.joblib")


def load_val_df() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "val.csv")


def load_train_df() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "train.csv")


def best_run_name(results_df: pd.DataFrame, model_name: str = None) -> str:
    """The best (by PR-AUC) run overall, or for one specific algorithm."""
    df = results_df if model_name is None else results_df[results_df["model"] == model_name]
    best_row = df.sort_values("pr_auc", ascending=False).iloc[0]
    return best_row["run"]
