#!/usr/bin/env python
"""Run the end-to-end Telco churn experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from ai_portfolio.data import build_preprocessor, load_telco_dataset
from ai_portfolio.evaluation import (
    classification_metrics,
    save_confusion_matrix,
    save_metrics_table,
    save_roc_curve,
)
from ai_portfolio.models import (
    RANDOM_STATE,
    build_model_pipelines,
    get_continuous_scores,
    parameter_spaces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Telco churn classifiers.")
    parser.add_argument("--data", required=True, help="Path to the raw Telco churn CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for generated outputs.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller search spaces for a faster smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_telco_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.20,
        stratify=bundle.y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(
        bundle.numeric_features,
        bundle.categorical_features,
    )
    models = build_model_pipelines(preprocessor)
    spaces = parameter_spaces(quick=args.quick)
    cv_splits = 3 if args.quick else 5
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: list[dict[str, object]] = []
    roc_scores: dict[str, object] = {}
    cv_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    # Baseline uses the same preprocessing contract for a fair, deployable artifact.
    baseline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = classification_metrics(y_test, baseline_pred)
    rows.append({"model": "dummy_baseline", **baseline_metrics})
    save_confusion_matrix(y_test, baseline_pred, "dummy_baseline", output_dir)

    for name, pipeline in models.items():
        print(f"Tuning {name} ({cv_splits}-fold CV)...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spaces[name],
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        fitted_models[name] = best
        pred = best.predict(X_test)
        score = get_continuous_scores(best, X_test)
        roc_scores[name] = score

        metrics = classification_metrics(y_test, pred, score)
        rows.append({"model": name, **metrics})
        save_confusion_matrix(y_test, pred, name, output_dir)

        best_idx = search.best_index_
        cv_rows.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_accuracy": search.cv_results_["mean_test_accuracy"][best_idx],
                "cv_precision": search.cv_results_["mean_test_precision"][best_idx],
                "cv_recall": search.cv_results_["mean_test_recall"][best_idx],
                "cv_f1": search.cv_results_["mean_test_f1"][best_idx],
                "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best_idx],
            }
        )

    metrics_path = save_metrics_table(rows, output_dir)
    pd.DataFrame(cv_rows).sort_values("cv_f1", ascending=False).to_csv(
        output_dir / "cross_validation_summary.csv",
        index=False,
    )
    save_roc_curve(y_test, roc_scores, output_dir)

    # Select using cross-validation only; the test set remains a final unbiased check.
    best_name = max(cv_rows, key=lambda row: row["cv_f1"])["model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, output_dir / "best_model.joblib")

    print(f"Best model by CV F1: {best_name}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {output_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
