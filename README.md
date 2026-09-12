# AI Implementation & Machine Learning Portfolio

A production-minded machine learning portfolio built from my **BEng Software Engineering** AI Implementation work. The repository turns coursework and lab exercises into a cleaner, reusable codebase that demonstrates data engineering, model development, evaluation, interpretability, testing, and CI.

## Flagship project — Telco Customer Churn

The main project predicts whether a telecom customer is likely to churn. It demonstrates an end-to-end supervised learning workflow:

- reproducible data loading and validation
- missing-value handling and categorical encoding
- leakage-safe preprocessing with `ColumnTransformer`
- stratified train/test splitting and cross-validation
- baseline modelling plus Logistic Regression, Random Forest, SVM, KNN, and MLP
- hyperparameter tuning with F1 as the refit metric
- accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves
- model persistence and reusable prediction code
- feature interpretation for tree and linear models
- automated tests and GitHub Actions CI

> The original coursework preprocessed the full dataset before modelling. This portfolio refactor moves preprocessing **inside each scikit-learn pipeline**, so transformations are learned only from training folds. That is a safer, more production-ready pattern and reduces data-leakage risk.

## Repository structure

```text
.
├── src/ai_portfolio/          # Reusable ML package
│   ├── data.py                # Loading, cleaning and preprocessing
│   ├── models.py              # Model registry and search spaces
│   └── evaluation.py          # Metrics and plots
├── scripts/
│   └── run_experiment.py      # End-to-end training/evaluation CLI
├── labs/
│   ├── lab02_preprocessing.py
│   ├── lab07_clustering.py
│   ├── lab08_decision_tree.py
│   └── lab10_mnist_neural_network.py
├── tests/                     # Fast unit tests
├── data/README.md             # Dataset setup and schema
├── docs/
│   ├── ARCHITECTURE.md
│   └── COURSEWORK_ORIGINS.md
├── .github/workflows/ci.yml
├── pyproject.toml
└── Makefile
```

## Quick start

```bash
git clone https://github.com/ibrahimm2106/AI-Implementation.git
cd AI-Implementation

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"
```

Place the Telco churn CSV at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then run the portfolio experiment:

```bash
python scripts/run_experiment.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir artifacts
```

For a faster smoke run:

```bash
python scripts/run_experiment.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir artifacts --quick
```

## Validated quick benchmark

The refactored pipeline was smoke-tested on the 7,043-row dataset with `random_state=42`, a stratified 80/20 split, and 3-fold cross-validation.

| Model | CV F1 | CV ROC-AUC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | **0.637** | 0.845 | **0.631** | 0.840 |
| Logistic Regression | 0.629 | **0.845** | 0.618 | **0.841** |
| KNN | 0.600 | 0.832 | 0.582 | 0.827 |
| SVM | 0.598 | 0.828 | 0.588 | 0.818 |
| MLP | 0.570 | 0.838 | 0.569 | 0.835 |
| Dummy baseline | — | — | 0.000 | — |

Random Forest was selected by cross-validation F1. Logistic Regression achieved a very similar ROC-AUC, showing why model choice should consider the business objective rather than accuracy alone.

## Engineering decisions

The codebase intentionally separates **data preparation**, **model construction**, and **evaluation**. This makes experiments easier to test and prevents notebook-only logic from becoming the source of truth.

The training pipeline handles missing values, scaling, and one-hot encoding inside cross-validation. Every candidate model uses the same split, scoring rules, and random seed where supported. The hold-out test set is reserved for final evaluation rather than model selection.

## Lab portfolio

The `labs/` directory contains cleaned, standalone versions of selected practical work:

| Lab | Topic | Skills shown |
|---|---|---|
| 02 | Data preprocessing | imputation, scaling, one-hot encoding, export |
| 07 | Clustering | Euclidean distance, K-means concepts, Ward hierarchical clustering, dendrograms |
| 08 | Decision trees | categorical encoding, supervised classification, interpretable rules |
| 10 | Neural networks | TensorFlow/Keras, MNIST, validation curves, multiclass evaluation |

The lab implementations were rewritten to remove machine-specific absolute paths and make them reusable from the command line.

## What this repository demonstrates to employers

- Python software engineering beyond a single notebook
- end-to-end ML workflow design
- reproducibility and configuration discipline
- awareness of leakage, class imbalance, and robust evaluation
- readable, modular code with type hints and docstrings
- automated testing and CI
- ability to explain model trade-offs and make implementation decisions

## Run tests and linting

```bash
pytest
ruff check .
```

Or:

```bash
make test
make lint
```

## Deep-learning lab

TensorFlow is optional so the core project stays lightweight:

```bash
pip install -e ".[deep-learning]"
python labs/lab10_mnist_neural_network.py --epochs 10 --output-dir artifacts/mnist
```

## Responsible ML notes

Customer churn prediction can influence retention offers and customer treatment. A deployable system should therefore consider data quality, subgroup performance, privacy, explainability, monitoring for drift, and human oversight. High predictive performance alone is not enough.

## Background

This repository was developed from university AI Implementation coursework and lab work, then refactored into a portfolio-quality engineering project. See [`docs/COURSEWORK_ORIGINS.md`](docs/COURSEWORK_ORIGINS.md) for the mapping from the learning material to the code in this repository.
