"""Coursework 2 source snapshot — Telco churn model comparison.

This public portfolio edition preserves the submitted experiment design while removing
machine-specific file handling. Exact result CSVs from the coursework are stored in
../outputs/.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv("telco_processed.csv").apply(pd.to_numeric, errors="coerce").dropna()
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

models = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000, random_state=RANDOM_STATE))]),
    "RandomForest": Pipeline([("scaler", "passthrough"), ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(class_weight="balanced", random_state=RANDOM_STATE))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "MLP": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=2000, early_stopping=True, random_state=RANDOM_STATE))]),
}

spaces = {
    "LogisticRegression": {"clf__C": [0.01, 0.1, 1.0, 10.0, 50.0]},
    "RandomForest": {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 5, 10, 20], "clf__min_samples_leaf": [1, 2, 4]},
    "SVM": {"clf__kernel": ["linear", "rbf"], "clf__C": [0.1, 1.0, 10.0]},
    "KNN": {"clf__n_neighbors": [5, 9, 15, 25, 35], "clf__weights": ["uniform", "distance"]},
    "MLP": {"clf__hidden_layer_sizes": [(50,), (100,), (100, 50)], "clf__activation": ["relu", "tanh"], "clf__alpha": [0.0001, 0.001, 0.01]},
}

results = []
for name, pipe in models.items():
    search = RandomizedSearchCV(pipe, spaces[name], n_iter=min(8, np.prod([len(v) for v in spaces[name].values()])), scoring=scoring, refit="f1", cv=cv, random_state=RANDOM_STATE, n_jobs=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    score = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else best.decision_function(X_test)
    results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, score)})

DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pd.DataFrame(results).to_csv("model_evaluation_results.csv", index=False)
