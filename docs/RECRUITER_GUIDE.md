# Recruiter / Interview Guide

This repository is designed to be understandable without opening every notebook.

## If you have 60 seconds

1. Read the root [`README.md`](../README.md).
2. Open [`src/ai_portfolio/data.py`](../src/ai_portfolio/data.py) to see leakage-aware preprocessing.
3. Open [`src/ai_portfolio/models.py`](../src/ai_portfolio/models.py) to see the model registry.
4. Open [`scripts/run_experiment.py`](../scripts/run_experiment.py) to see experiment orchestration.
5. Scan [`tests/`](../tests/) and [`.github/workflows/`](../.github/workflows/) for engineering quality.
6. Review [`RESULTS.md`](RESULTS.md) for evidence.

## If you have 5 minutes

Review the two coursework pages:

- [`coursework/coursework-1/README.md`](../coursework/coursework-1/README.md) — data preparation and visual analysis.
- [`coursework/coursework-2/README.md`](../coursework/coursework-2/README.md) — model comparison, tuning, evaluation and interpretability.

Then inspect [`labs/README.md`](../labs/README.md) for breadth across clustering, decision trees and neural networks.

## Discussion points for a software / ML interview

### 1. Why keep academic evidence and refactored modules?

The coursework/lab evidence preserves the learning process and outputs. The modules show how the same ideas can be organised into reusable, testable software with separated concerns.

### 2. What was improved after the coursework?

- machine-specific paths removed from portfolio scripts
- preprocessing moved into model pipelines where appropriate
- reusable package structure added
- command-line training and prediction added
- tests added
- CI added
- Docker/Makefile added
- model card and architecture docs added
- result provenance retained through source notebooks, result CSVs and rendered portfolio visuals

### 3. Why F1 and ROC-AUC instead of accuracy alone?

The churn class is the minority. A majority-class baseline can achieve reasonable accuracy while identifying no churners. Precision, recall, F1 and ROC-AUC make the error trade-offs visible.

### 4. Why is preprocessing inside the pipeline important?

When scaling, imputation and encoding are learned inside each training fold, validation data does not influence the transformation parameters. This reduces leakage risk and makes the experiment closer to real deployment behaviour.

### 5. What would be needed for production deployment?

- stronger data contracts and validation
- subgroup/fairness analysis
- threshold selection based on business cost
- monitoring and drift detection
- access controls and privacy review
- model registry/versioning
- stronger observability
- API/service layer if real-time scoring were required

## Role alignment

### Graduate / Junior Software Engineer

Evidence: modular Python, tests, CLI design, CI, Docker, documentation, code organisation.

### Graduate / Junior Data or ML Engineer

Evidence: preprocessing, reproducible experiments, cross-validation, model evaluation, interpretation and artifact generation.

### Roles asking for stronger engineering maturity

Evidence: separation of concerns, pipelines, tests, CI, reproducibility, responsible-AI documentation and the distinction between exploratory notebooks and reusable application code.

The repository demonstrates engineering practices; it does not claim that academic work alone is equivalent to commercial mid-level experience.
