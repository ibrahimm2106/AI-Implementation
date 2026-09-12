# Coursework 1 — Telco Customer Churn Data Preparation

[![Notebook](https://img.shields.io/badge/Original-Notebook-F37626?logo=jupyter&logoColor=white)](original/Coursework_1_Telco_Preprocessing.ipynb)
[![Report](https://img.shields.io/badge/Report-Markdown-083fa1?logo=markdown&logoColor=white)](report/Coursework_1_Report.md)
[![Outputs](https://img.shields.io/badge/Outputs-Visuals%20%2B%20CSV-2DA44E)](../../docs/RESULTS.md)

Coursework 1 establishes the data-preparation stage of the Telco churn project. The original submission uses Python, pandas, NumPy, Matplotlib, Seaborn and scikit-learn to inspect, clean, transform, visualise and export the data for later machine-learning work.

## Original workflow

1. Load the Telco Customer Churn dataset.
2. Inspect shape, data types, missing values and duplicates.
3. Convert `TotalCharges` to numeric.
4. Fill missing `TotalCharges` values using the median.
5. Remove duplicate rows.
6. Drop `customerID` as a non-predictive identifier.
7. Encode binary Yes/No columns.
8. One-hot encode multi-class categorical features.
9. Standardise `tenure`, `MonthlyCharges` and `TotalCharges`.
10. Produce descriptive statistics and visualisations.
11. Export the processed dataset for Coursework 2.

## Original code snapshot

![Coursework 1 code](../../docs/images/code/coursework1_preprocessing_code.svg)

## Original output — churn distribution

![Churn distribution](../../docs/images/portfolio/cw1_churn_distribution.svg)

## Files preserved

```text
coursework-1/
├── original/
│   ├── Coursework_1_Telco_Preprocessing.ipynb
│   └── Coursework_1_Telco_Preprocessing.py
├── report/
│   └── Coursework_1_Report.md
└── data/
    └── telco_summary_statistics.csv
```

The original notebook is kept as the primary executable coursework artefact. The report is published as a GitHub-friendly Markdown edition, and the repository uses lightweight SVG evidence on the main pages so reviewers do not need to download large binary files.

## Portfolio engineering improvement

The original coursework performs preprocessing on the complete dataset because its focus was data preparation. The portfolio package under [`../../src/ai_portfolio/`](../../src/ai_portfolio/) improves the later modelling workflow by moving learned transformations such as imputation, scaling and encoding **inside scikit-learn pipelines**, so they are fitted within the training folds during cross-validation.

This preserves the original academic work while demonstrating a stronger leakage-aware production pattern.

## Academic note

This directory preserves original coursework evidence for portfolio provenance. It is not provided as a template for other students to submit.
