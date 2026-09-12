# Assessment-to-Repository Mapping

This page explains how the portfolio evidence maps to the original module tasks without republishing the university assessment briefs themselves.

## Coursework 1 — data preparation and processing

The coursework focused on selecting a dataset/project theme and preparing data for future model training.

| Requirement area | Repository evidence |
|---|---|
| Project overview / ML problem | Telco Customer Churn project in the Coursework 1 report and root README |
| Data understanding | original notebook inspection, summary statistics and visualisations |
| Missing/inconsistent data | `TotalCharges` conversion + median handling |
| Duplicate / irrelevant attributes | duplicate removal + `customerID` removal |
| Encoding | binary mapping + one-hot encoding |
| Scaling | `StandardScaler` on numeric features |
| Visualisations | churn distribution and documented exploratory outputs |
| Processed dataset | Coursework 1 processed-data evidence + `data/sample_telco.csv` for a lightweight public sample |
| Reproducible code | original notebook/source plus portfolio refactor |

## Coursework 2 — machine-learning models and evaluation

The coursework required multiple supervised models, tuning, robust validation, suitable evaluation metrics, comparison/critical analysis and ethical analysis.

| Requirement area | Repository evidence |
|---|---|
| 3+ models | LR, Random Forest, SVM, KNN and MLP |
| Hyperparameter tuning | `RandomizedSearchCV` search spaces in the original notebook |
| K-fold validation | 3-fold `StratifiedKFold` in the submitted experiment |
| Accuracy / Precision / Recall / F1 | generated CSV result tables |
| ROC / AUC | documented model comparison and final ROC-AUC values |
| Confusion matrices | final selected-model visual + notebook evidence |
| Model comparison | tuned CV table and hold-out table |
| Interpretability | RF feature importance + LR coefficients |
| Threshold analysis | original notebook/report analysis |
| Calibration | Brier score + original notebook/report analysis |
| Ethical analysis | Coursework 2 report + `MODEL_CARD.md` |

## Practical lab mapping

| Lab | Main topic | Portfolio evidence |
|---|---|---|
| Lab 02 | preprocessing | notebook/source, processed-data evidence, reusable preprocessing script |
| Lab 07 | K-means / hierarchical clustering | notebook/source, Ward dendrogram, reusable script |
| Lab 08 | decision trees | notebook/source, sample predictions, exported tree logic, reusable script |
| Lab 10 | neural networks | notebook/source, MNIST metrics, learning/prediction visuals, reusable TensorFlow script |
