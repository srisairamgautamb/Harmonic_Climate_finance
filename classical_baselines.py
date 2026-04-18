"""
Phase H1 -- Classical Baselines

Linear Spectral Regression, RBF-SVM, Random Forest, LSTM (PyTorch),
and VAR forecast baselines.  Unified comparison table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from statsmodels.tsa.vector_ar.var_model import VAR

import config
from utils import classification_metrics, regression_metrics

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


def train_linear_spectral_regression(
    PHI_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    PHI_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
) -> dict:
    """Ridge regression + Logistic regression on PCA-reduced spectral features."""
    pca = PCA(n_components=50, random_state=config.SEED)
    X_train = pca.fit_transform(PHI_train)
    X_test = pca.transform(PHI_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_reg)
    y_pred_reg = ridge.predict(X_test)
    reg_met = regression_metrics(y_test_reg, y_pred_reg)

    lr = LogisticRegression(max_iter=500, random_state=config.SEED)
    lr.fit(X_train, y_train_cls)
    y_pred_cls = lr.predict(X_test)
    y_prob_cls = lr.predict_proba(X_test)[:, 1] if len(np.unique(y_train_cls)) > 1 else None

    cls_met = classification_metrics(y_test_cls, y_pred_cls, y_prob_cls)

    result = {**reg_met, **{f"cls_{k}": v for k, v in cls_met.items()}}
    result["_y_true"] = y_test_reg.tolist()
    result["_y_pred"] = y_pred_reg.tolist()
    return result


def train_rbf_svm(
    PHI_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    PHI_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
) -> dict:
    """RBF-SVM regression and classification."""
    pca = PCA(n_components=50, random_state=config.SEED)
    X_train = pca.fit_transform(PHI_train)
    X_test = pca.transform(PHI_test)

    svr = SVR(kernel="rbf", C=10.0, gamma="scale")
    svr.fit(X_train, y_train_reg)
    y_pred_reg = svr.predict(X_test)
    reg_met = regression_metrics(y_test_reg, y_pred_reg)

    svc = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=config.SEED)
    svc.fit(X_train, y_train_cls)
    y_pred_cls = svc.predict(X_test)
    y_prob_cls = svc.predict_proba(X_test)[:, 1] if len(np.unique(y_train_cls)) > 1 else None
    cls_met = classification_metrics(y_test_cls, y_pred_cls, y_prob_cls)

    return {**reg_met, **{f"cls_{k}": v for k, v in cls_met.items()}}


def train_random_forest(
    PHI_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    PHI_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
) -> dict:
    """Random Forest regression."""
    pca = PCA(n_components=50, random_state=config.SEED)
    X_train = pca.fit_transform(PHI_train)
    X_test = pca.transform(PHI_test)

    rf = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=config.SEED)
    rf.fit(X_train, y_train_reg)
    y_pred = rf.predict(X_test)
    return regression_metrics(y_test_reg, y_pred)


class _LSTMModel(nn.Module):
    """Two-layer LSTM for time-series regression."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(
    PHI_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    PHI_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
) -> dict:
    """LSTM regression on raw spectral features."""
    seq_len = 1
    input_size = PHI_train.shape[1]
    output_size = 1

    X_tr = torch.tensor(PHI_train, dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train_reg, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(PHI_test, dtype=torch.float32).unsqueeze(1)

    model = _LSTMModel(input_size, 64, output_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(100):
        optimiser.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimiser.step()
        if (epoch + 1) % 25 == 0:
            logger.info("  LSTM epoch %d: loss=%.4f", epoch + 1, loss.item())

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).numpy().flatten()

    out_dir = Path(config.OUTPUTS_DIR) / "phase_H"
    torch.save(model.state_dict(), out_dir / "lstm_model.pt")
    np.savez(out_dir / "lstm_predictions.npz", y_true=y_test_reg, y_pred=y_pred)

    return regression_metrics(y_test_reg, y_pred)


def train_var_forecast(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    q_fin: int,
) -> dict:
    """VAR 1-step-ahead forecast on test set."""
    model = VAR(Z_train)
    try:
        lag_order = model.select_order(maxlags=config.VAR_MAX_LAG).aic
        if lag_order < 1:
            lag_order = 1
    except Exception:
        lag_order = 2
    var_fit = model.fit(lag_order)

    n_test = Z_test.shape[0]
    d = Z_train.shape[1]
    preds = np.zeros((n_test, d))
    history = Z_train.copy()

    for t in range(n_test):
        fc = var_fit.forecast(history[-lag_order:], steps=1)
        preds[t] = fc[0]
        history = np.vstack([history, Z_test[t : t + 1]])

    y_true_fin = Z_test[:, -q_fin:]
    y_pred_fin = preds[:, -q_fin:]
    vix_idx = 0

    out_dir = Path(config.OUTPUTS_DIR) / "phase_H"
    np.savez(
        out_dir / "var_predictions.npz",
        y_true=y_true_fin[:, vix_idx],
        y_pred=y_pred_fin[:, vix_idx],
    )

    return regression_metrics(y_true_fin[:, vix_idx], y_pred_fin[:, vix_idx])


def compile_comparison_table() -> pd.DataFrame:
    """Load all results and compile a single comparison CSV."""
    out_base = Path(config.OUTPUTS_DIR)
    rows = []

    for phase, model_name, fname in [
        ("phase_H", "LSR", "lsr_results.json"),
        ("phase_H", "RBF-SVM", "rbf_svm_results.json"),
        ("phase_H", "RF", "rf_results.json"),
        ("phase_H", "LSTM", "lstm_results.json"),
        ("phase_H", "VAR", "var_results.json"),
    ]:
        fpath = out_base / phase / fname
        if fpath.exists():
            with open(fpath) as fh:
                d = json.load(fh)
            rows.append({
                "model": model_name,
                "rmse_vix": d.get("rmse", None),
                "r2_vix": d.get("r2", None),
                "f1_stress": d.get("f1", d.get("cls_f1", None)),
                "auc_stress": d.get("auc_roc", d.get("cls_auc_roc", None)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_base / "phase_H" / "model_comparison.csv", index=False)
    return df


def run_classical_baselines() -> None:
    """Phase H1 entry point."""
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    out_dir = Path(config.OUTPUTS_DIR) / "phase_H"
    out_dir.mkdir(parents=True, exist_ok=True)

    spectral = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_B" / "spectral_features.npz"),
        allow_pickle=True,
    )
    PHI = spectral["PHI"]
    window_centers = spectral["window_centers"].astype(int)

    with open(proc_dir / "split_indices.json") as fh:
        split = json.load(fh)

    train_mask = window_centers < split["train_end"]
    test_mask = window_centers >= split["train_end"]
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(test_idx) == 0:
        test_idx = train_idx[-max(1, len(train_idx) // 5) :]
        train_idx = train_idx[: -len(test_idx)]

    F_hat = pd.read_csv(proc_dir / "F_hat.csv", index_col=0, parse_dates=True)
    vix = F_hat.iloc[:, 0].values
    vix_wc = vix[window_centers[window_centers < len(vix)]]
    if len(vix_wc) < len(window_centers):
        vix_wc = np.pad(vix_wc, (0, len(window_centers) - len(vix_wc)), mode="edge")

    y_vix_next = np.roll(vix_wc, -1)
    
    # Compute proper rolling classification targets to match HQG
    vix_mean = pd.Series(vix_wc).rolling(12, min_periods=1).mean().values
    vix_std = pd.Series(vix_wc).rolling(12, min_periods=1).std().fillna(1.0).values
    y_stress = (vix_wc > vix_mean + vix_std).astype(int)
    
    PHI_train = PHI[train_idx]
    PHI_test = PHI[test_idx]
    
    y_train_reg = y_vix_next[train_idx]
    y_test_reg = y_vix_next[test_idx]
    y_train_cls = y_stress[train_idx]
    y_test_cls = y_stress[test_idx]

    for name, fn in [
        ("LSR", train_linear_spectral_regression),
        ("RBF-SVM", train_rbf_svm),
        ("RF", train_random_forest),
        ("LSTM", train_lstm),
    ]:
        logger.info("Training %s ...", name)
        result = fn(PHI_train, y_train_reg, y_train_cls, PHI_test, y_test_reg, y_test_cls)
        # Save predictions for DM test (LSR is used as the best-classical benchmark)
        y_true_list = result.pop("_y_true", None)
        y_pred_list = result.pop("_y_pred", None)
        if y_true_list is not None:
            slug = name.lower().replace("-", "_")
            np.savez(out_dir / f"{slug}_predictions.npz",
                     y_true=np.array(y_true_list), y_pred=np.array(y_pred_list))
        fname = name.lower().replace("-", "_") + "_results.json"
        with open(out_dir / fname, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info("  %s: %s", name, result)

    Z_hat = pd.read_csv(proc_dir / "Z_hat.csv", index_col=0, parse_dates=True)
    Z_vals = Z_hat.values
    q_fin = len(config.FINANCIAL_RISK_VARS)

    # CRITICAL FIX: Align VAR to the same window-center test indices as HQG.
    # Use window_centers to index Z_hat so all 7 models share an identical test set.
    wc_all = window_centers.astype(int)
    safe_wc = np.clip(wc_all, 0, len(Z_vals) - 1)
    Z_wc = Z_vals[safe_wc]  # shape [n_windows, d]

    Z_train_wc = Z_wc[train_idx]   # [n_train_windows, d]
    Z_test_wc  = Z_wc[test_idx]    # [n_test_windows, d]  -- same 52 as HQG

    logger.info("Training VAR forecast on aligned windows (train=%d, test=%d)...",
                len(train_idx), len(test_idx))
    var_result = train_var_forecast(Z_train_wc, Z_test_wc, q_fin)
    with open(out_dir / "var_results.json", "w") as fh:
        json.dump(var_result, fh, indent=2)
    logger.info("  VAR: %s", var_result)

    compile_comparison_table()
    logger.info("Phase H complete.")

