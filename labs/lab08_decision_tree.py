"""Lab 08 — decision-tree classification with categorical data."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to PastHires.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "Hired" not in df.columns:
        raise ValueError("Expected a target column named 'Hired'.")

    X = df.drop(columns=["Hired"])
    y = df["Hired"]

    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("tree", DecisionTreeClassifier(random_state=42)),
        ]
    )
    model.fit(X, y)

    samples = pd.DataFrame(
        [
            {
                "Years Experience": 10,
                "Employed?": "Y",
                "Previous employers": 4,
                "Level of Education": "BS",
                "Top-tier school": "N",
                "Interned": "N",
            },
            {
                "Years Experience": 1,
                "Employed?": "N",
                "Previous employers": 4,
                "Level of Education": "BS",
                "Top-tier school": "N",
                "Interned": "N",
            },
        ]
    )

    samples = samples.reindex(columns=X.columns)
    print("Predictions:", model.predict(samples).tolist())

    encoded_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    print(export_text(model.named_steps["tree"], feature_names=list(encoded_feature_names)))


if __name__ == "__main__":
    main()
