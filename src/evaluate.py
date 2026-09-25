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


def evaluate_cv(model, X, y, groups, n_splits=5, return_folds=False, val_transform=None):
    """Stratified-group 5-fold CV, grouped by cycle (per the notebook 03 split decision).

    `model` should be a full pipeline (preprocessing + classifier), not a bare
    classifier, so each fold fits its own imputer/scaler on that fold's training
    data only — reusing a pipeline pre-fit on the whole train set would leak
    validation-fold statistics into the imputer/scaler. Returns a dict of
    mean/std for PR-AUC, recall, precision, F1, balanced accuracy, and ROC-AUC.

    Optional: `return_folds=True` adds the per-fold scores under "folds";
    `val_transform` is applied to each validation fold before prediction (never
    to the training fold), e.g. to simulate a feature missing at serving time.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = {name: [] for name in METRIC_NAMES}

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        if val_transform is not None:
            X_val = val_transform(X_val)

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
    if return_folds:
        result["folds"] = scores
    return result


def find_runs(mask, cycle):
    """Inclusive (start, end) index pairs of maximal runs of True in `mask` that stay within one cycle."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1] and cycle[j + 1] == cycle[i]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def episode_metrics(y_true, y_pred, cycle, early_horizon=5, rows_per_hour=3600):
    """Event-level view of row-level predictions, for time-ordered rows (about 1 reading per second).

    A stop episode is a run of consecutive stop rows; an alarm is a run of consecutive flagged rows.
    An alarm is useful if it overlaps an episode or starts within `early_horizon` rows before one
    (an early warning); every other alarm is a false-alarm run.
    """
    y = np.asarray(y_true) == 1
    p = np.asarray(y_pred).astype(bool)
    c = np.asarray(cycle)
    episodes, alarms = find_runs(y, c), find_runs(p, c)

    def lead_start(s):
        lo = s
        while lo > max(s - early_horizon, 0) and c[lo - 1] == c[s]:
            lo -= 1
        return lo

    n_ep = len(episodes)
    during = sum(bool(p[s:e + 1].any()) for s, e in episodes)
    onset = sum(bool(p[s]) for s, _ in episodes)
    before = sum(bool(p[lead_start(s):s].any()) for s, _ in episodes)
    caught_or_warned = sum(bool(p[lead_start(s):e + 1].any()) for s, e in episodes)
    windows = [(lead_start(s), e) for s, e in episodes]
    false_alarms = sum(not any(a <= we and b >= ws for ws, we in windows) for a, b in alarms)
    hours = len(y) / rows_per_hour
    return {
        "stop episodes": n_ep,
        "caught during (%)": 100 * during / n_ep,
        "onset row flagged (%)": 100 * onset / n_ep,
        f"warned in {early_horizon} rows before (%)": 100 * before / n_ep,
        "caught or warned (%)": 100 * caught_or_warned / n_ep,
        "alarm runs": len(alarms),
        "false-alarm runs": false_alarms,
        "false-alarm runs per hour": false_alarms / hours,
    }
