"""Small IO helpers the dashboard reads from the artifacts training
already produced — no retraining or recomputation happens here.

Deployment note: the dashboard only ever needs val.csv in full (the
threshold tuner scores real feature rows against it) — train.csv and
test.csv are only used here for small aggregate stats and a ~150-row
sample, both precomputed once by `src/prepare_deploy_artifacts.py` into
`results/dataset_summary.json` and `results/live_sim_sequence.csv`.
That's what keeps the deployable artifact set to ~20MB instead of
needing the full (150MB+) processed data folder. See
`scripts/package_deploy_bundle.py`."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
RESULTS_PATH = Path("results/model_comparison.csv")
SHAP_CACHE_DIR = Path("results/shap_cache")
EMBEDDING_CACHE_DIR = Path("results/embedding_cache")
PROCESSED_DIR = Path("data/processed")
DATASET_SUMMARY_PATH = Path("results/dataset_summary.json")
LIVE_SIM_SEQUENCE_PATH = Path("results/live_sim_sequence.csv")


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


def load_test_df() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "test.csv")


def best_run_name(results_df: pd.DataFrame, model_name: str = None) -> str:
    """The best (by PR-AUC) run overall, or for one specific algorithm."""
    df = results_df if model_name is None else results_df[results_df["model"] == model_name]
    best_row = df.sort_values("pr_auc", ascending=False).iloc[0]
    return best_row["run"]


def default_fraud_row_position(shap_cache: dict, val_df: pd.DataFrame) -> int:
    """Position (not original row index) of the first fraud case in a
    SHAP cache's sample — a more interesting default to show than
    whatever happens to be in position 0."""
    class_labels = val_df.loc[shap_cache["row_index"], "Class"].to_numpy()
    fraud_positions = np.where(class_labels == 1)[0]
    return int(fraud_positions[0]) if len(fraud_positions) else 0


def load_dataset_summary() -> dict:
    """Small precomputed counts (train/val/test size + fraud count, and
    fraud count by hour) — everything dataset_overview_card and
    timeline_card need, without loading the full train/test CSVs."""
    with open(DATASET_SUMMARY_PATH) as f:
        return json.load(f)


def load_live_sim_sequence() -> pd.DataFrame:
    """The precomputed live-simulation replay sequence — see
    `src/prepare_deploy_artifacts.py` for how it's built."""
    return pd.read_csv(LIVE_SIM_SEQUENCE_PATH)
