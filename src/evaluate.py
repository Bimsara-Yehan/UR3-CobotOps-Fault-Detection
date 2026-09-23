"""Shared cross-validation harness and metric reporting used by all model notebooks."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RANDOM_STATE

METRIC_NAMES = ["pr_auc", "recall", "precision", "f1", "balanced_accuracy", "roc_auc"]


def evaluate_cv(model, X, y, groups, n_splits=5):
    """Stratified-group 5-fold CV, grouped by cycle (per the notebook 03 split decision).

    `model` should be a full pipeline (preprocessing + classifier), not a bare
    classifier, so each fold fits its own imputer/scaler on that fold's training
    data only — reusing a pipeline pre-fit on the whole train set would leak
    validation-fold statistics into the imputer/scaler. Returns a dict of
    mean/std for PR-AUC, recall, precision, F1, balanced accuracy, and ROC-AUC.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = {name: [] for name in METRIC_NAMES}

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_val)
        y_proba = fold_model.predict_proba(X_val)[:, 1]

        scores["pr_auc"].append(average_precision_score(y_val, y_proba))
        scores["recall"].append(recall_score(y_val, y_pred))
        scores["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        scores["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_val, y_pred))
        scores["roc_auc"].append(roc_auc_score(y_val, y_proba))

    result = {}
    for name in METRIC_NAMES:
        result[f"{name}_mean"] = float(np.mean(scores[name]))
        result[f"{name}_std"] = float(np.std(scores[name]))
    return result
