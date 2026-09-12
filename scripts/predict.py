#!/usr/bin/env python
"""Generate churn predictions from a saved portfolio model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn for customer rows in a CSV file.")
    parser.add_argument("--model", required=True, help="Path to best_model.joblib")
    parser.add_argument("--input", required=True, help="CSV containing raw customer features")
    parser.add_argument("--output", default="predictions.csv", help="Prediction output CSV")
    args = parser.parse_args()

    model = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    features = frame.drop(columns=["customerID", "Churn"], errors="ignore").copy()
    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    predictions = model.predict(features)
    result = frame.copy()
    result["predicted_churn"] = predictions

    if hasattr(model, "predict_proba"):
        result["churn_probability"] = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        result["churn_score"] = model.decision_function(features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions to {output_path}")


if __name__ == "__main__":
    main()
