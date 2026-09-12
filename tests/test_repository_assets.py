import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_evidence_files_exist():
    required = [
        "coursework/coursework-1/README.md",
        "coursework/coursework-2/outputs/final_results_summary.csv",
        "docs/images/portfolio/cw1_churn_distribution.svg",
        "docs/images/portfolio/cw2_confusion_matrix.svg",
        "labs/README.md",
        "MODEL_CARD.md",
    ]
    for relative in required:
        assert (REPO_ROOT / relative).exists(), relative


def test_final_coursework2_metrics_are_parseable():
    path = REPO_ROOT / "coursework/coursework-2/outputs/final_test_metrics_best_model.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["BestModel"] == "RandomForest"
    assert float(rows[0]["Test ROC AUC"]) > 0.8
