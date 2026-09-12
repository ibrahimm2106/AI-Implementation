"""Model registry and tuning spaces."""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


RANDOM_STATE = 42


def build_model_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Build candidate model pipelines that share identical preprocessing."""

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    SVC(
                        class_weight="balanced",
                        probability=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "mlp": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    MLPClassifier(
                        early_stopping=True,
                        max_iter=1500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def parameter_spaces(quick: bool = False) -> Mapping[str, dict[str, list[object]]]:
    """Return compact, meaningful search spaces for each candidate model."""

    if quick:
        return {
            "logistic_regression": {"model__C": [0.1, 1.0]},
            "random_forest": {
                "model__n_estimators": [200],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 4],
            },
            "svm": {
                "model__C": [0.5, 1.0],
                "model__kernel": ["linear"],
            },
            "knn": {
                "model__n_neighbors": [9, 25],
                "model__weights": ["uniform"],
            },
            "mlp": {
                "model__hidden_layer_sizes": [(50,)],
                "model__alpha": [0.0001, 0.001],
            },
        }

    return {
        "logistic_regression": {"model__C": [0.01, 0.1, 1.0, 10.0, 50.0]},
        "random_forest": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "svm": {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"],
        },
        "knn": {
            "model__n_neighbors": [5, 9, 15, 25, 35],
            "model__weights": ["uniform", "distance"],
        },
        "mlp": {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
        },
    }


def get_continuous_scores(model: BaseEstimator, X):
    """Return probabilities or decision-function scores for ROC/AUC calculations."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None
