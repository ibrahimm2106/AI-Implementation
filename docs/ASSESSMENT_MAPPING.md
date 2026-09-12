# Assessment-to-Repository Mapping

This page explains how the portfolio evidence maps to the original module tasks without republishing the university assessment briefs themselves.

## Coursework 1 — data preparation and processing

The coursework focused on selecting a dataset/project theme and preparing data for future model training.

| Requirement area | Repository evidence |
|---|---|
| Project overview / ML problem | Telco Customer Churn project in the Coursework 1 report and root README |
| Data understanding | notebook/source edition, summary statistics and visualisations |
| Missing/inconsistent data | `TotalCharges` conversion + median handling |
| Duplicate / irrelevant attributes | duplicate removal + `customerID` removal |
| Encoding | binary mapping + one-hot encoding |
| Scaling | `StandardScaler` on numeric features |
| Visualisations | churn distribution and documented exploratory outputs |
| Processed dataset | preprocessing/export logic plus Coursework 1 summary-statistics evidence; large duplicate datasets are not republished |
| Reproducible code | coursework source edition plus leakage-aware portfolio refactor |

## Coursework 2 — machine-learning models and evaluation

The coursework required multiple supervised models, tuning, robust validation, suitable evaluation metrics, comparison/critical analysis and ethical analysis.

| Requirement area | Repository evidence |
|---|---|
| 3+ models | Logistic Regression, Random Forest, SVM, KNN and MLP |
| Hyperparameter tuning | `RandomizedSearchCV` search spaces in the coursework source edition |
| K-fold validation | 3-fold `StratifiedKFold` in the recorded experiment |
| Accuracy / Precision / Recall / F1 | preserved CSV result tables |
| ROC / AUC | documented model comparison and final ROC-AUC values |
| Confusion matrices | final selected-model visual + report evidence |
| Model comparison | tuned CV table and hold-out table |
| Interpretability | RF feature importance + LR coefficients |
| Threshold analysis | coursework report analysis |
| Calibration | Brier score + coursework report analysis |
| Ethical analysis | Coursework 2 report + `MODEL_CARD.md` |

## Practical lab mapping

| Lab | Main topic | Portfolio evidence |
|---|---|---|
| Lab 02 | preprocessing | notebook/source edition + reusable preprocessing script + documented processed-data output |
| Lab 07 | K-means / hierarchical clustering | notebook/source edition, Ward dendrogram, reusable script |
| Lab 08 | decision trees | notebook/source edition, sample predictions, tree logic, reusable script |
| Lab 10 | neural networks | notebook/source edition, MNIST metrics, learning/prediction visuals, reusable TensorFlow script |
