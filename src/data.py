"""Load the raw Kaggle creditcard.csv, add a few features, and split it
into train/val/test sets for the rest of the project to use."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")

SECONDS_PER_DAY = 24 * 60 * 60


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from "
            "https://www.kaggle.com/mlg-ulb/creditcardfraud and place it there "
            "(see data/raw/README.md)."
        )
    return pd.read_csv(path)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time is seconds elapsed since the first transaction, spanning ~2 days."""
    df["hour_of_day"] = (df["Time"] % SECONDS_PER_DAY) // 3600
    df["day"] = (df["Time"] // SECONDS_PER_DAY).astype(int)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """Stratified train/val/test split so the rare fraud class stays
    proportionally represented in every split."""
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["Class"], random_state=random_state
    )
    # val_size is a fraction of the full dataset, so rescale it relative
    # to what's left in train_val_df.
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["Class"],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def build_explain_sample(df: pd.DataFrame, n_legit=2000, random_state=42) -> pd.DataFrame:
    """All fraud rows plus a random sample of legit rows — small enough
    for SHAP/t-SNE/UMAP to run on quickly, with enough fraud cases in
    the mix to be useful for the dashboard to explore."""
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0].sample(n=n_legit, random_state=random_state)
    return pd.concat([fraud_df, legit_df]).sort_index()


def prepare_data(raw_path: Path = RAW_PATH, processed_dir: Path = PROCESSED_DIR):
    df = load_raw_data(raw_path)
    df = add_time_features(df)
    df = add_amount_features(df)

    train_df, val_df, test_df = split_data(df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fraud_count = split_df["Class"].sum()
        fraud_rate = fraud_count / len(split_df) * 100
        print(
            f"{name}: {len(split_df):,} rows, {fraud_count} fraud "
            f"({fraud_rate:.3f}%)"
        )

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_data()
