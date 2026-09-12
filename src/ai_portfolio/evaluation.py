"""Evaluation helpers for binary classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Compute consistent binary-classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    return metrics


def save_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str | Path) -> Path:
    """Save a confusion-matrix figure and return its path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{model_name}.png"

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_roc_curve(y_true, scores: dict[str, object], output_dir: str | Path) -> Path:
    """Save a single ROC comparison plot for all models with continuous scores."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "roc_curves.png"

    plt.figure(figsize=(8, 6))
    for name, score in scores.items():
        if score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_value = roc_auc_score(y_true, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Hold-out Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_metrics_table(rows: list[dict[str, object]], output_dir: str | Path) -> Path:
    """Save evaluation rows as a CSV sorted by hold-out F1."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_metrics.csv"

    frame = pd.DataFrame(rows).sort_values("f1", ascending=False)
    frame.to_csv(path, index=False)
    return path
