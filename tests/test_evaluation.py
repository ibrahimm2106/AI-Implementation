from ai_portfolio.evaluation import classification_metrics


def test_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.6, 0.7, 0.9]

    metrics = classification_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
