"""Lab 02 — reusable preprocessing example.

Refactored from the original notebook exercise so paths are supplied at runtime and the
workflow can be reused on different tabular datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path) if input_path.suffix.lower() == ".csv" else pd.read_excel(input_path)

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]

    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )

    transformed = transformer.fit_transform(df)
    result = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="artifacts/lab02_processed.csv")
    args = parser.parse_args()

    result = preprocess(Path(args.input), Path(args.output))
    print(result.head())
    print(f"Saved {len(result)} processed rows to {args.output}")


if __name__ == "__main__":
    main()
