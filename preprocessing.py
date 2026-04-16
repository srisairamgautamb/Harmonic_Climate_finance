"""
Phase A4 -- Preprocessing Module

Stationarity testing (ADF/KPSS), seasonal and regular differencing,
standardisation, and train/test split.

Incorporates Moderate Issue 6 fix: after seasonal differencing the
ADF/KPSS re-test uses ``regression="c"`` instead of ``"ct"``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss

import config
from utils import assert_shape, log_shape

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)


def run_stationarity_tests(
    series: pd.Series,
    var_name: str,
    regression: str = "ct",
) -> dict:
    """ADF and KPSS stationarity tests for a single variable."""
    try:
<<<<<<< HEAD
        adf_result = adfuller(series.dropna(), maxlag=12, regression=regression)
=======
        adf_result = adfuller(series.dropna(), maxlags=12, regression=regression)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
        adf_stat = float(adf_result[0])
        adf_pval = float(adf_result[1])
        adf_crit = {k: float(v) for k, v in adf_result[4].items()}
    except Exception as exc:
        logger.warning("ADF failed for %s: %s", var_name, exc)
        adf_stat, adf_pval, adf_crit = np.nan, 1.0, {}

    try:
        kpss_stat_val, kpss_pval_val, _, kpss_crit = kpss(
            series.dropna(), regression=regression, nlags="auto"
        )
        kpss_stat = float(kpss_stat_val)
        kpss_pval = float(kpss_pval_val)
    except Exception as exc:
        logger.warning("KPSS failed for %s: %s", var_name, exc)
        kpss_stat, kpss_pval = np.nan, 0.0

    adf_stationary = adf_pval < 0.05
    kpss_stationary = kpss_pval > 0.05

    return {
        "variable": var_name,
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "adf_crit": adf_crit,
        "adf_stationary": adf_stationary,
        "kpss_stat": kpss_stat,
        "kpss_pval": kpss_pval,
        "kpss_stationary": kpss_stationary,
    }


def _detect_seasonality(series: pd.Series, lag: int = 12) -> bool:
    """Simple seasonality detection via autocorrelation at *lag*."""
    try:
        from statsmodels.tsa.stattools import acf

        ac = acf(series.dropna(), nlags=lag + 1, fft=True)
        return abs(ac[lag]) > 0.3
    except Exception:
        return False


def transform_to_stationary(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Apply seasonal/regular differencing to achieve stationarity."""
    recipe: dict[str, dict] = {}
    report_rows: list[dict] = []
    df_out = pd.DataFrame(index=df.index)

    all_vars = list(df.columns)
    block_map: dict[str, str] = {}
    for v in config.DRIVER_VARS:
        block_map[v] = "driver"
    for v in config.CLIMATE_VARS:
        block_map[v] = "climate"
    for v in config.CLIMATE_RISK_VARS:
        block_map[v] = "climate_risk"
    for v in config.FINANCIAL_RISK_VARS:
        block_map[v] = "financial_risk"

    for var in all_vars:
        s = df[var].dropna().copy()
        seasonal_diff = False
        n_diffs = 0

        has_season = _detect_seasonality(s)
        if has_season:
            s = s.diff(12).dropna()
            seasonal_diff = True
            logger.info("%s: seasonal difference applied (lag=12)", var)

        test_reg = "c" if seasonal_diff else "ct"
        result = run_stationarity_tests(s, var, regression=test_reg)
        adf_ok = result["adf_stationary"]
        kpss_ok = result["kpss_stationary"]

        while not (adf_ok and kpss_ok) and n_diffs < 2:
            s = s.diff(1).dropna()
            n_diffs += 1
            result = run_stationarity_tests(s, var, regression="c")
            adf_ok = result["adf_stationary"]
            kpss_ok = result["kpss_stationary"]

        final_status = "stationary" if (adf_ok and kpss_ok) else "nonstationary"
        if final_status == "nonstationary":
            logger.warning("%s: still nonstationary after %d diffs", var, n_diffs)

        recipe[var] = {
            "log": False,
            "seasonal_diff": seasonal_diff,
            "n_diffs": n_diffs,
        }
        report_rows.append({
            "variable": var,
            "block": block_map.get(var, "unknown"),
            "adf_stat": result["adf_stat"],
            "adf_pval": result["adf_pval"],
            "kpss_stat": result["kpss_stat"],
            "kpss_pval": result["kpss_pval"],
            "n_diffs_applied": n_diffs,
            "seasonal_diff_applied": seasonal_diff,
            "final_status": final_status,
        })
        df_out[var] = s.reindex(df.index)

    df_out = df_out.dropna(how="all")
    df_out = df_out.ffill().bfill()

    proc_dir = Path(config.DATA_PROCESSED_DIR)
    pd.DataFrame(report_rows).to_csv(
        proc_dir / "stationarity_report.csv", index=False
    )
    with open(proc_dir / "transform_recipe.json", "w") as fh:
        json.dump(recipe, fh, indent=2)
    logger.info("Stationarity report saved. Transform recipe saved.")

    return df_out, recipe


def standardize(
    df_train: pd.DataFrame,
    df_full: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on *df_train*, transform *df_full*."""
    scaler = StandardScaler()
    scaler.fit(df_train.values)
    scaled = pd.DataFrame(
        scaler.transform(df_full.values),
        index=df_full.index,
        columns=df_full.columns,
    )
    return scaled, scaler


def preprocess_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline: load, test, transform, split, save."""
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        proc_dir / "aligned_monthly.csv", index_col=0, parse_dates=True
    )
    logger.info("Loaded aligned data: shape=%s", df.shape)

    D_hat = df[config.DRIVER_VARS]
    C_hat = df[config.CLIMATE_VARS]
    R_cl_hat = df[config.CLIMATE_RISK_VARS]
    R_fin_hat = df[config.FINANCIAL_RISK_VARS]

    log_shape(D_hat.values, "D_hat (raw)")
    log_shape(C_hat.values, "C_hat (raw)")
    log_shape(R_cl_hat.values, "R_cl_hat (raw)")
    log_shape(R_fin_hat.values, "R_fin_hat (raw)")

    df_stationary, recipe = transform_to_stationary(df)

    T_prime = len(df_stationary)
    train_end_idx = int(0.8 * T_prime)

    df_train = df_stationary.iloc[:train_end_idx]
    df_scaled, scaler = standardize(df_train, df_stationary)

    D_hat = df_scaled[config.DRIVER_VARS]
    C_hat = df_scaled[config.CLIMATE_VARS]
    R_cl_hat = df_scaled[config.CLIMATE_RISK_VARS]
    R_fin_hat = df_scaled[config.FINANCIAL_RISK_VARS]

    Z_hat = pd.concat([D_hat, C_hat, R_cl_hat, R_fin_hat], axis=1)
    U_hat = pd.concat([D_hat, C_hat, R_cl_hat], axis=1)
    F_hat = R_fin_hat.copy()

    log_shape(Z_hat.values, "Z_hat")
    log_shape(U_hat.values, "U_hat")
    log_shape(F_hat.values, "F_hat")

    Z_hat.to_csv(proc_dir / "Z_hat.csv")
    U_hat.to_csv(proc_dir / "U_hat.csv")
    F_hat.to_csv(proc_dir / "F_hat.csv")

    joblib.dump(scaler, proc_dir / "scaler.pkl")

    split = {
        "train_start": 0,
        "train_end": train_end_idx,
        "test_start": train_end_idx,
        "test_end": T_prime,
    }
    with open(proc_dir / "split_indices.json", "w") as fh:
        json.dump(split, fh, indent=2)

    logger.info(
        "Preprocessing complete. T'=%d, train=%d, test=%d",
        T_prime, train_end_idx, T_prime - train_end_idx,
    )
    return Z_hat, U_hat, F_hat
