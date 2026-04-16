"""
Shared utilities for the Spectral Climate-Financial Risk Transmission
pipeline.  Custom exceptions, shape assertions, numerical helpers,
metrics, and I/O utilities.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Custom Exceptions ───────────────────────────────────────────────────

class DataUnavailableError(Exception):
    """Raised when a required dataset cannot be downloaded or located."""


class DimensionError(Exception):
    """Raised when a matrix or array has an unexpected shape."""


class NumericalInstabilityError(Exception):
    """Raised when a matrix operation fails due to numerical issues."""


# ── Shape Assertion ─────────────────────────────────────────────────────

def assert_shape(
    arr: np.ndarray,
    expected: tuple[int | None, ...],
    name: str,
) -> None:
    """Assert *arr* matches *expected* shape.  ``None`` skips a dimension."""
    if len(arr.shape) != len(expected):
        raise DimensionError(
            f"{name}: expected {len(expected)}D array, got "
            f"{len(arr.shape)}D. Shape: {arr.shape}"
        )
    for i, (actual, exp) in enumerate(zip(arr.shape, expected)):
        if exp is not None and actual != exp:
            raise DimensionError(
                f"{name}: dimension {i} expected {exp}, got {actual}. "
                f"Full shape: {arr.shape}"
            )


# ── Positive Definiteness ──────────────────────────────────────────────

def ensure_positive_definite(
    K: np.ndarray,
    jitter: float = 1e-6,
    name: str = "matrix",
) -> np.ndarray:
    """Symmetrise and add minimal jitter so *K* is PD."""
    K = (K + K.T) / 2.0
    eigvals = np.linalg.eigvalsh(K)
    min_eig = float(eigvals.min())
    if min_eig < jitter:
        correction = jitter - min_eig + 1e-8
        K = K + correction * np.eye(K.shape[0])
        logger.info(
            "%s: added jitter %.2e (min eigenvalue was %.2e)",
            name, correction, min_eig,
        )
    return K


# ── Logging Helpers ─────────────────────────────────────────────────────

def log_shape(arr: np.ndarray, name: str) -> None:
    """Log shape, dtype, min, and max of *arr*."""
    logger.info(
        "  %s: shape=%s, dtype=%s, min=%.4f, max=%.4f",
        name, arr.shape, arr.dtype, float(arr.min()), float(arr.max()),
    )


# ── NumPy Archive I/O ──────────────────────────────────────────────────

def save_npz(path: str, **arrays: Any) -> None:
    """Save arrays to *.npz* and log each shape."""
    for name, arr in arrays.items():
        if hasattr(arr, "shape"):
            log_shape(arr, name)
    np.savez(path, **arrays)
    logger.info("Saved: %s", path)


def load_npz(path: str) -> dict[str, np.ndarray]:
    """Load *.npz* and return as plain dict."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ── Time-Series Cross-Validation ───────────────────────────────────────

def time_series_cv_splits(
    n: int,
    n_splits: int = 5,
    min_train_size: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-series CV splits."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    test_size = (n - min_train_size) // n_splits
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = min(train_end + test_size, n)
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


# ── Median Heuristic for Kernel Bandwidth ──────────────────────────────

def median_heuristic(X: np.ndarray) -> float:
    """Compute median pairwise Euclidean distance (sub-sampled)."""
    from sklearn.metrics import pairwise_distances

    n = X.shape[0]
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    dists = pairwise_distances(X_sub, metric="euclidean")
    upper = dists[np.triu_indices_from(dists, k=1)]
    return float(np.median(upper))


# ── Diebold-Mariano Test ───────────────────────────────────────────────

def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
) -> dict[str, float | bool]:
    """Two-sided Diebold-Mariano test for equal predictive accuracy."""
    import scipy.stats as stats

    d = errors_1 ** 2 - errors_2 ** 2
    T = len(d)
    d_bar = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=1))
    gamma_sum = 0.0
    if h > 1:
        for lag in range(1, h):
            gamma_sum += (1.0 - lag / (h + 1)) * float(
                np.cov(d[lag:], d[:-lag])[0, 1]
            )
    var_d = (gamma_0 + 2.0 * gamma_sum) / T
    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else 0.0
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return {
        "DM_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 4),
        "reject_H0_at_5pct": bool(p_value < 0.05),
    }


# ── Metrics ─────────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """RMSE, MAE, R-squared, directional accuracy."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    dir_acc = float(np.mean(dir_true == dir_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "directional_accuracy": dir_acc}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, and optionally AUC-ROC."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
<<<<<<< HEAD
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = None
=======
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    return metrics
