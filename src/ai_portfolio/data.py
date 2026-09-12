"""Data loading and preprocessing for the Telco churn project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass(frozen=True)
class DatasetBundle:
    """Features, target, and detected column groups."""

    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def load_telco_dataset(path: str | Path) -> DatasetBundle:
    """Load and minimally clean the Telco churn dataset.

    Cleaning is intentionally limited before train/test splitting. Transformations that
    learn from data (imputation, scaling, encoding) are performed inside the model pipeline
    to reduce data-leakage risk during cross-validation.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = {TARGET_COLUMN, "TotalCharges", "tenure", "MonthlyCharges"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad = sorted(df.loc[y.isna(), TARGET_COLUMN].astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values in {TARGET_COLUMN!r}: {bad}")

    X = df.drop(columns=[TARGET_COLUMN])

    # Treat SeniorCitizen as categorical because it is a binary indicator, not a magnitude.
    numeric_features = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in X.columns]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return DatasetBundle(
        X=X,
        y=y.astype(int),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create leakage-safe preprocessing for numeric and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
