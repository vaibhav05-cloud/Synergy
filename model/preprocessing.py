"""
preprocessing.py
-----------------
Turns raw deployment metadata (the columns in train.csv / validation.csv / test.csv)
into the numeric feature matrix the model actually trains on.

This file is imported by BOTH:
  1. train_model.py (during training)
  2. the FastAPI backend later (during live prediction)
so that a deployment coming in from GitHub goes through the EXACT same
transformation as the training data did. This consistency is important —
if training and live prediction preprocess data differently, the model's
predictions become unreliable.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---- Column groups -----------------------------------------------------
# These names must exactly match the columns in train.csv / validation.csv / test.csv

TARGET_COL = "risk_level"

CATEGORICAL_COLS = [
    "repo_domain",
    "candidate_type",
    "primary_service",
    "branch_type",
    "workflow_kind",
]

# Every other column (except the target and the categorical ones above) is numeric.
def get_numeric_cols(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c not in CATEGORICAL_COLS + [TARGET_COL]
    ]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a handful of 'combination' features. Trees CAN learn interactions
    on their own, but giving them ready-made ones usually helps accuracy,
    especially with a small-ish dataset like ours (~1000 training rows).

    Each new feature encodes a real-world risk story:
      - off_hours_large_change: a big change AND it's happening off-hours
      - critical_service_migration: a database migration AND the service is critical
      - failure_prone_critical_service: this service already fails a lot AND it's critical
    """
    df = df.copy()
    df["off_hours_large_change"] = df["is_off_hours"] * df["changed_files"]
    df["critical_service_migration"] = df["service_criticality"] * df["is_database_migration"]
    df["failure_prone_critical_service"] = df["past_failure_rate_30d"] * df["service_criticality"]
    return df


def build_preprocessor(numeric_cols: list) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer:
      - categorical columns -> one-hot encoded
      - numeric columns     -> passed through unchanged (tree models don't need scaling)

    handle_unknown="ignore" is important: if the live GitHub data later has a
    category value we never saw during training (e.g. a brand-new service name),
    this stops the pipeline from crashing — it just encodes it as "all zeros".
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return preprocessor


def prepare_features(df: pd.DataFrame):
    """
    One call that does everything: add interaction features, then split
    into X (features) and y (target, if present).
    Returns (X, y, numeric_cols) — numeric_cols is needed to build the preprocessor.
    """
    df = add_interaction_features(df)
    numeric_cols = get_numeric_cols(df)

    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df[CATEGORICAL_COLS + numeric_cols]

    return X, y, numeric_cols