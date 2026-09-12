# Coursework and lab origins

This repository is a professional refactor of the AI Implementation work completed during a BEng Software Engineering module.

## Coursework 1 — data preparation

The first coursework focused on defining an ML problem, understanding a dataset, cleaning data, handling missing values, encoding categorical features, scaling numeric features, visualising patterns, and exporting a processed dataset.

The original Telco churn notebook performed those steps successfully, but used machine-specific Windows paths and transformed the complete dataset before later modelling. The portfolio version keeps the useful cleaning logic while moving learned transformations into training pipelines.

## Coursework 2 — modelling and evaluation

The second coursework required at least three supervised models, parameter tuning, k-fold cross-validation, suitable classification metrics, visual comparisons, critical analysis, and an ethical discussion.

The submitted work went further by experimenting with:

- Logistic Regression
- Random Forest
- SVM
- KNN
- MLP neural network
- a dummy baseline
- confusion matrices and ROC curves
- feature importance and linear coefficients
- threshold analysis
- probability calibration and Brier score

Those ideas are preserved in the portfolio architecture, but organised into reusable Python modules rather than one large notebook.

## Lab 02 — preprocessing

Used as the basis for the standalone preprocessing example covering missing-value assessment, median/constant imputation, feature scaling, one-hot encoding, and processed-data export.

## Lab 07 — clustering

The learning material introduced Euclidean distance, K-means, centroids, and hierarchical clustering. The exercise used ten 2D points with Ward linkage and a dendrogram. The portfolio script keeps that exercise and adds a compact K-means comparison.

## Lab 08 — decision trees

The practical task used a hiring dataset, categorical encoding, a `DecisionTreeClassifier`, two sample predictions, and readable tree rules. The portfolio version removes hard-coded local paths and wraps preprocessing and classification into a reusable pipeline.

## Lab 10 — neural networks

The lab introduced TensorFlow/Keras tensors, flatten and dense layers, ReLU, softmax, MNIST training, validation curves, predictions, confusion matrices, and inspection of hidden-layer representations.

The portfolio version provides a clean MNIST training script and makes TensorFlow an optional dependency so the main repository stays lightweight.

## Public-repository hygiene

University assessment briefs, marked reports, submission ZIPs, student identifiers, absolute local file paths, and notebook checkpoint files are intentionally not published. They were used as source material for the refactor, while the public repository contains the reusable engineering work.
