# Coursework 2 — Original Output Artefacts

This directory preserves the complete set of **21 non-checkpoint files** found in the original `Outputs/` folder of the uploaded Coursework 2 submission archive. Filenames are kept unchanged for provenance.

## Model comparison and baseline evaluation

- [`model_evaluation_results.csv`](model_evaluation_results.csv) — hold-out accuracy, precision, recall, F1 and AUC for Logistic Regression, SVM, KNN, Random Forest, MLP and the Dummy baseline.
- [`roc_curves_all_models.png`](roc_curves_all_models.png) — ROC comparison across the evaluated models.
- [`conf_matrix_LogisticRegression.png`](conf_matrix_LogisticRegression.png)
- [`conf_matrix_RandomForest.png`](conf_matrix_RandomForest.png)
- [`conf_matrix_SVM.png`](conf_matrix_SVM.png)
- [`conf_matrix_KNN.png`](conf_matrix_KNN.png)
- [`conf_matrix_MLP.png`](conf_matrix_MLP.png)
- [`conf_matrix_Dummy.png`](conf_matrix_Dummy.png)

## Hyperparameter tuning and final model selection

- [`tuned_model_results_fast.csv`](tuned_model_results_fast.csv) — best parameter settings and stratified cross-validation results for the tuned models.
- [`final_results_summary.csv`](final_results_summary.csv) — combined summary of tuned results and final selected-model test results.
- [`final_test_metrics_best_model.csv`](final_test_metrics_best_model.csv) — final Random Forest hold-out metrics.
- [`final_confusion_matrix_best_model.png`](final_confusion_matrix_best_model.png) — final selected-model confusion matrix.
- [`final_roc_best_model.png`](final_roc_best_model.png) — final selected-model ROC curve.

## Calibration

- [`brier_score_best_model.csv`](brier_score_best_model.csv) — Brier score for the selected Random Forest model.
- [`calibration_curve_best_model.png`](calibration_curve_best_model.png) — probability calibration plot.

## Interpretability

- [`rf_feature_importance.csv`](rf_feature_importance.csv) — Random Forest feature-importance values.
- [`rf_feature_importance.png`](rf_feature_importance.png) — feature-importance visualisation.
- [`lr_coefficients.csv`](lr_coefficients.csv) — Logistic Regression coefficients and absolute magnitudes.
- [`lr_coefficients.png`](lr_coefficients.png) — Logistic Regression coefficient visualisation.

## Logistic Regression threshold analysis

- [`lr_precision_recall_curve.png`](lr_precision_recall_curve.png) — precision-recall curve used for threshold analysis.
- [`lr_threshold_table.csv`](lr_threshold_table.csv) — complete threshold, precision, recall and F1 table from the coursework experiment.

## Recorded final result

The selected Random Forest model records approximately **0.7884 accuracy**, **0.6418 precision**, **0.4599 recall**, **0.5358 F1**, **0.8226 ROC-AUC** and a **0.1454 Brier score** in the preserved output files.

These artefacts are retained as coursework evidence. The separate portfolio engineering code in the repository may use safer/refactored implementation patterns, but these files remain the original recorded outputs.
