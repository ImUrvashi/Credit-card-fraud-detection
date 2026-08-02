"""Small IO helpers the dashboard reads from the artifacts training
already produced — no retraining or recomputation happens here.

Deployment note: if `deploy_artifacts/` exists, every path below points
there instead of the full local `data/`/`models/`/`results/` folders.
`deploy.sh` builds that directory from a full local run (`./run.sh`) —
it holds only the single best model (generically named `model.joblib`,
since which algorithm wins can change between retrains) plus val.csv
and the small SHAP/embedding/summary caches, not the full 58-model,
780MB local pipeline output. It's small enough (~20MB) to commit and
push directly, so a git-based host just needs to pull and start the
app — no separate upload/download step. See UNDERSTANDING.md §14."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEPLOY_DIR = Path("deploy_artifacts")
# Prefer the full local pipeline output when it's present, even if
# deploy_artifacts/ also happens to exist (e.g. right after running
# ./deploy.sh in a full local checkout) - only fall back to the small
# deployed subset when the full output genuinely isn't there, which is
# the real production case (a git clone with just deploy_artifacts/).
IS_DEPLOYED = DEPLOY_DIR.exists() and not Path("results/model_comparison.csv").exists()

if IS_DEPLOYED:
    MODELS_DIR = DEPLOY_DIR
    RESULTS_PATH = DEPLOY_DIR / "model_comparison.csv"
    SHAP_CACHE_DIR = DEPLOY_DIR / "shap_cache"
    EMBEDDING_CACHE_DIR = DEPLOY_DIR / "embedding_cache"
    VAL_PATH = DEPLOY_DIR / "val.csv"
    DATASET_SUMMARY_PATH = DEPLOY_DIR / "dataset_summary.json"
    LIVE_SIM_SEQUENCE_PATH = DEPLOY_DIR / "live_sim_sequence.csv"
else:
    MODELS_DIR = Path("models")
    RESULTS_PATH = Path("results/model_comparison.csv")
    SHAP_CACHE_DIR = Path("results/shap_cache")
    EMBEDDING_CACHE_DIR = Path("results/embedding_cache")
    VAL_PATH = Path("data/processed/val.csv")
    DATASET_SUMMARY_PATH = Path("results/dataset_summary.json")
    LIVE_SIM_SEQUENCE_PATH = Path("results/live_sim_sequence.csv")

PROCESSED_DIR = Path("data/processed")  # local-only: train.csv/test.csv live here


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["run"] = df["model"] + "__" + df["strategy"]
    return df


def load_model(run_name: str):
    """In deployed mode there's only ever one model on disk (whichever
    was best when `deploy.sh` last ran), saved as a fixed filename since
    `run_name` itself can change between retrains — `run_name` is
    ignored in that case, not looked up."""
    if IS_DEPLOYED:
        return joblib.load(MODELS_DIR / "model.joblib")
    return joblib.load(MODELS_DIR / f"{run_name}.joblib")


def load_shap_cache(model_name: str) -> dict:
    return joblib.load(SHAP_CACHE_DIR / f"{model_name}.joblib")


def load_embedding_cache(method: str) -> dict:
    return joblib.load(EMBEDDING_CACHE_DIR / f"{method}.joblib")


def load_val_df() -> pd.DataFrame:
    return pd.read_csv(VAL_PATH)


def load_train_df() -> pd.DataFrame:
    """Local-only — never called in deployed mode (see dataset_summary.json)."""
    return pd.read_csv(PROCESSED_DIR / "train.csv")


def load_test_df() -> pd.DataFrame:
    """Local-only — never called in deployed mode (see live_sim_sequence.csv)."""
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
