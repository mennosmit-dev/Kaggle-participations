"""
Shared utilities for the House Prices Kaggle experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REMOVE_COLUMNS = [
    "Alley",
    "MasVnrArea",
    "FireplaceQu",
    "PoolQC",
    "Fence",
    "MiscFeature",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def transform_base(df: pd.DataFrame, *, drop_id: bool = False) -> pd.DataFrame:
    """Apply the minimal column cleanup used across experiments."""
    df = df.copy()
    df = df.drop(columns=REMOVE_COLUMNS, errors="ignore")
    if drop_id and "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def split_features_target(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split train data into X and y, using log1p on the target."""
    df = transform_base(train_df, drop_id=False)
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for mixed numeric/categorical data."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def prepare_test_features(test_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return test ids and transformed test features."""
    test_df = transform_base(test_df, drop_id=False)
    test_ids = test_df["Id"].copy()
    X_test = test_df.drop(columns=["Id"], errors="ignore")
    return test_ids, X_test


def save_submission(ids: pd.Series, pred_log: np.ndarray, output_path: str | Path) -> None:
    """Save Kaggle submission with inverse-transformed predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "Id": ids,
            "SalePrice": np.expm1(pred_log),
        }
    )
    submission.to_csv(output_path, index=False)
