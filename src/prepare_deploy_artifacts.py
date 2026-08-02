"""Precomputes small summaries of train.csv/test.csv so the dashboard
never needs to load those full files — only the aggregate stats and a
~150-row sample it actually uses. Run once locally after src.train;
writes results/dataset_summary.json and results/live_sim_sequence.csv,
both tiny and safe to ship with a deployment.
"""

import json
from pathlib import Path

import pandas as pd

from src.data import build_explain_sample

PROCESSED_DIR = Path("data/processed")
SUMMARY_PATH = Path("results/dataset_summary.json")
LIVE_SIM_SEQUENCE_PATH = Path("results/live_sim_sequence.csv")

HOURS = range(24)


def split_stats(df: pd.DataFrame) -> dict:
    return {"count": int(len(df)), "fraud": int(df["Class"].sum())}


def build_summary(train_df, val_df, test_df) -> dict:
    fraud_by_hour = train_df[train_df["Class"] == 1].groupby("hour_of_day").size()
    fraud_by_hour = fraud_by_hour.reindex(HOURS, fill_value=0)
    return {
        "train": split_stats(train_df),
        "val": split_stats(val_df),
        "test": split_stats(test_df),
        "fraud_by_hour": {str(h): int(c) for h, c in fraud_by_hour.items()},
    }


def build_live_sim_sequence(test_df, n_legit=60, random_state=7) -> pd.DataFrame:
    """A small chronological slice of held-out test transactions for the
    live-simulation demo. Oversamples fraud relative to its real ~0.17%
    rate — a faithful random walk through test_df would run for a very
    long time between fraud hits, which makes for a bad demo."""
    sample = build_explain_sample(test_df, n_legit=n_legit, random_state=random_state)
    return sample.sort_values("Time").reset_index(drop=True)


def run_all():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(build_summary(train_df, val_df, test_df), f, indent=2)
    print(f"wrote {SUMMARY_PATH}")

    build_live_sim_sequence(test_df).to_csv(LIVE_SIM_SEQUENCE_PATH, index=False)
    print(f"wrote {LIVE_SIM_SEQUENCE_PATH}")


if __name__ == "__main__":
    run_all()
