"""Precompute 2D PCA/t-SNE/UMAP projections of a transaction sample, so
the dashboard's embedding scatter can just switch between cached arrays
instead of re-running these (t-SNE/UMAP are too slow to run live)."""

from pathlib import Path

import joblib
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data import build_explain_sample
from src.explain import FULL_SAMPLE_LEGIT
from src.train import FEATURE_COLS

MODELS_DIR = Path("models")
RESULTS_PATH = Path("results/model_comparison.csv")
EMBEDDING_CACHE_DIR = Path("results/embedding_cache")

RANDOM_STATE = 42


# Automatically load the best model from all the experiments we ran earlier.
def best_overall_model():
    results_df = pd.read_csv(RESULTS_PATH)
    best_row = results_df.sort_values("pr_auc", ascending=False).iloc[0]
    run_name = f"{best_row['model']}__{best_row['strategy']}"
    return joblib.load(MODELS_DIR / f"{run_name}.joblib")


def run_all():
    val_df = pd.read_csv("data/processed/val.csv")  # reads val data since we want to visualize on unseen data.
    sample_df = build_explain_sample(val_df, n_legit=FULL_SAMPLE_LEGIT)

    X = sample_df[FEATURE_COLS]
    y = sample_df["Class"].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    model = best_overall_model()
    pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

    projections = {
        "pca": PCA(n_components=2, random_state=RANDOM_STATE),
        "tsne": TSNE(n_components=2, random_state=RANDOM_STATE, init="pca"),
        "umap": umap.UMAP(n_components=2, random_state=RANDOM_STATE),
    }

    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for name, projector in projections.items():
        print(f"computing {name} embedding ...")
        coords = projector.fit_transform(X_scaled)
        joblib.dump(
            {
                "coords": coords,
                "y": y,
                "row_index": sample_df.index.to_numpy(),
                "pred_proba": pred_proba,
            },
            EMBEDDING_CACHE_DIR / f"{name}.joblib",
        )


if __name__ == "__main__":
    run_all()
