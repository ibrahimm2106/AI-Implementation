# Coursework 2 Report — Portfolio Text Edition

> Based on the submitted coursework report. Personal student identifiers are omitted from this public edition.

## Introduction

This project predicts Telco customer churn as a binary classification task using the cleaned dataset from Coursework 1. The modelling dataset contained **7,038 usable rows and 31 features**, with approximately **26.53% churn**.

## Experimental setup

Five model families were compared: Logistic Regression, Random Forest, SVM, KNN and MLP, alongside a majority-class dummy baseline. The submitted experiment used an 80/20 stratified train/test split, 3-fold `StratifiedKFold`, hyperparameter search and F1 as the primary refit metric. Accuracy, precision, recall, F1 and ROC-AUC were retained for comparison.

## Main findings

Random Forest achieved the strongest tuned CV F1 (**0.6344**) and ROC-AUC (**0.8451**), narrowly ahead of Logistic Regression on F1. SVM achieved the highest recall but at lower precision. The final selected Random Forest test artifact reported accuracy **0.7884**, precision **0.6418**, recall **0.4599**, F1 **0.5358** and ROC-AUC **0.8226**.

## Interpretation

The strongest Random Forest importances included `TotalCharges`, `tenure`, `MonthlyCharges`, two-year contract status, fibre-optic internet and electronic-check payment. Logistic Regression coefficients were also inspected to provide a more transparent linear view of feature effects. These are predictive associations, not causal explanations.

## Thresholds and calibration

The coursework explored precision-recall threshold trade-offs and probability calibration. The final selected model's Brier score was **0.1454**. In a real retention system, threshold choice should be driven by the relative business cost of false negatives and false positives.

## Ethical analysis

Customer-level prediction requires careful treatment of privacy, fairness, transparency and automated decision-making risk. A real deployment should include subgroup error analysis, access controls, human review, monitoring and clear limits on how predictions affect customers.

## Reflection

The project demonstrates that model selection on imbalanced data cannot rely on accuracy alone. It also shows the trade-off between predictive performance, interpretability and computational cost, and why experiment provenance matters when comparing cross-validation and final test outputs.
