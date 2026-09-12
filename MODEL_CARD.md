# Model Card — Telco Customer Churn Portfolio

## Summary

This repository contains an academic-to-portfolio machine-learning project for predicting whether a telecom customer is likely to churn. It is designed to demonstrate model development and software-engineering practices, not to operate as an automated production decision system.

The original Coursework 2 experiment compares Logistic Regression, Random Forest, SVM, KNN and MLP classifiers plus a dummy baseline. The portfolio refactor also provides reusable training/evaluation code around the same problem.

## Data

The original Telco Customer Churn dataset used in Coursework 1 contains **7,043 rows and 21 columns**. Inputs include demographic/service indicators, tenure, contract/payment information, internet/support services, `MonthlyCharges` and `TotalCharges`.

The target is `Churn`.

## Original experiment

Coursework 2 used a processed numeric dataset and reported **7,038 usable rows and 31 features** after its loading/conversion step. The target distribution was approximately:

- 73.47% no churn
- 26.53% churn

This imbalance is why the project considers precision, recall, F1 and ROC-AUC rather than relying on accuracy alone.

## Model selection

The original tuned experiment used stratified cross-validation and selected the winner using F1.

| Model | CV F1 | CV ROC-AUC |
|---|---:|---:|
| Random Forest | **0.6344** | **0.8451** |
| Logistic Regression | 0.6294 | 0.8450 |
| SVM | 0.5960 | 0.8392 |
| MLP | 0.5946 | 0.8390 |
| KNN | 0.5824 | 0.8217 |

Random Forest was selected by CV F1. The final best-model artifact reports:

| Metric | Result |
|---|---:|
| Accuracy | 0.7884 |
| Precision | 0.6418 |
| Recall | 0.4599 |
| F1 | 0.5358 |
| ROC-AUC | 0.8226 |
| Brier score | 0.1454 |

## Portfolio preprocessing

The reusable package uses a leakage-aware pattern:

- `customerID` is excluded from modelling.
- `TotalCharges` is converted to numeric.
- numeric values are median-imputed and standardised.
- categorical values are imputed and one-hot encoded.
- learned preprocessing lives inside the scikit-learn pipeline so cross-validation folds learn transformations only from their training data.

## Intended use

Appropriate uses:

- software / machine-learning portfolio review
- educational experimentation with binary classification
- comparing model-evaluation methods
- prototyping retention analytics with human review

## Not appropriate without further work

- fully automated customer treatment or pricing decisions
- high-stakes decisions affecting essential access or services
- deployment to a different customer population without validation
- causal claims about why a customer churns

## Key risks and limitations

- The dataset is historical and may not represent current customer behaviour.
- Feature importance and coefficients show predictive association, not causation.
- Subgroup fairness has not been fully audited.
- Customer/account information can be sensitive and requires appropriate data governance.
- Threshold choice should reflect the real cost of false positives and false negatives.
- Performance can degrade as products, pricing and customer behaviour change.

## Human oversight and monitoring

A production system should include human review, privacy controls, subgroup error analysis, drift monitoring, versioned model/data artefacts, threshold/cost analysis and clear explanations for how predictions are used.
