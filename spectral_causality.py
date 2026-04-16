"""
Phase C1 -- Spectral Granger Causality Module

VAR-based analytical spectral density, Geweke (1982) frequency-domain
Granger causality, partial spectral coherence, and phase lead-lag.

Incorporates Moderate Issue 4 fix: Hermitian spectral matrices use
``np.linalg.slogdet`` on real parts for the log-determinant ratio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

import config
from utils import assert_shape, log_shape, save_npz

<<<<<<< HEAD
from scipy.signal import find_peaks

=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
logger = logging.getLogger(__name__)

np.random.seed(config.SEED)


<<<<<<< HEAD
def detect_harmonic_families(S_Z: np.ndarray, freqs: np.ndarray) -> dict:
    """Detect harmonic families from average power spectrum."""
    if S_Z.ndim == 4:
        S_avg = np.mean(S_Z, axis=0) # [n_freqs, d, d]
    else:
        S_avg = S_Z

    power = np.real(np.trace(S_avg, axis1=1, axis2=2))
    power = power / np.max(power)

    peaks, _ = find_peaks(power, prominence=0.1, distance=5)
    if len(peaks) == 0:
        peaks = np.argsort(power)[-3:]
        
    nu = freqs[peaks]
    
    harmonics = []
    for r in nu:
        for m in [1, 2, 3]:
            if m * r <= np.pi:
                harmonics.append(float(m * r))
                
    combinations = []
    for i in range(len(nu)):
        for j in range(i + 1, len(nu)):
            if nu[i] + nu[j] <= np.pi:
                combinations.append(float(nu[i] + nu[j]))
            if np.abs(nu[i] - nu[j]) > 0:
                combinations.append(float(np.abs(nu[i] - nu[j])))
                
    all_freqs = set(np.round(harmonics + combinations, 5))
    
    return {
        "peaks_hz": nu.tolist(),
        "harmonics_hz": sorted(list(all_freqs)),
    }



=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
def fit_var_model(
    Z_train: np.ndarray,
    max_lags: int,
) -> object:
    """Fit VAR model on training data, selecting lag via AIC."""
    try:
<<<<<<< HEAD
        Z_use = Z_train.copy()
        model = VAR(Z_use)
        try:
            lag_order = model.select_order(maxlags=max_lags)
            selected_lag = lag_order.aic
            if selected_lag < 1:
                selected_lag = 1
        except Exception as e:
            logger.warning("VAR select_order failed (%s), defaulting to 1", e)
            selected_lag = 1

        try:
            var_fit = model.fit(selected_lag)
        except np.linalg.LinAlgError:
            logger.warning("VAR fit LinAlgError. Adding jitter and retrying.")
            Z_use += np.random.randn(*Z_use.shape) * 1e-5
            model = VAR(Z_use)
            var_fit = model.fit(1)
            selected_lag = 1

        logger.info("VAR selected lag: %d", selected_lag)
=======
        model = VAR(Z_train)
        lag_order = model.select_order(maxlags=max_lags)
        selected_lag = lag_order.aic
        if selected_lag < 1:
            selected_lag = 1
        logger.info("VAR selected lag: %d (AIC=%.2f)", selected_lag, lag_order.aic)
        var_fit = model.fit(selected_lag)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
        return var_fit
    except Exception as exc:
        logger.error("VAR fitting failed: %s", exc)
        raise


def compute_var_spectral_density(
    var_fit: object,
    n_freqs: int = 500,
) -> np.ndarray:
    """Compute analytical spectral density from VAR coefficients.

    Returns S_Z of shape [n_freqs, d, d] (complex).
    """
    try:
        coefs = var_fit.coefs
        sigma_u = var_fit.sigma_u
        p = coefs.shape[0]
        d = coefs.shape[1]

        freqs = np.linspace(0, np.pi, n_freqs)
        S_Z = np.zeros((n_freqs, d, d), dtype=complex)
        I_d = np.eye(d)

        for fi, omega in enumerate(freqs):
            A_omega = I_d.copy().astype(complex)
            for k in range(p):
                A_omega -= coefs[k] * np.exp(-1j * (k + 1) * omega)
            A_inv = np.linalg.inv(A_omega)
            S_Z[fi] = (1.0 / (2.0 * np.pi)) * A_inv @ sigma_u @ A_inv.conj().T

        log_shape(S_Z.real, "S_Z_VAR (real)")
        return S_Z
    except Exception as exc:
        logger.error("VAR spectral density computation failed: %s", exc)
        raise


def geweke_spectral_gc(
    S_Z: np.ndarray,
    x_idx: list[int],
    y_idx: list[int],
    freqs: np.ndarray,
) -> np.ndarray:
    """Geweke frequency-domain Granger causality X -> Y.

    Uses Schur complement with ``slogdet`` on real parts for
    Hermitian spectral matrices.
    """
    n_freqs = len(freqs)
    GC = np.zeros(n_freqs)

    for fi in range(n_freqs):
        S = S_Z[fi]
        S_YY = S[np.ix_(y_idx, y_idx)]
        S_XX = S[np.ix_(x_idx, x_idx)]
        S_YX = S[np.ix_(y_idx, x_idx)]
        S_XY = S[np.ix_(x_idx, y_idx)]

        try:
<<<<<<< HEAD
            # Apply slight diagonal stabilisation for singular/collinear series
            jitter = 1e-6 * np.eye(len(x_idx))
            jitter_y = 1e-6 * np.eye(len(y_idx))
            
            S_YY_given_X = S_YY - S_YX @ np.linalg.solve(S_XX + jitter, S_XY)

            sign_full, log_det_full = np.linalg.slogdet(S_YY.real + jitter_y)
            sign_cond, log_det_cond = np.linalg.slogdet(S_YY_given_X.real + jitter_y)

            if sign_full > 0 and sign_cond > 0 and np.isfinite(log_det_full) and np.isfinite(log_det_cond):
                gc_val = log_det_full - log_det_cond
                GC[fi] = max(gc_val, 0.0)
            else:
                GC[fi] = 0.0
=======
            S_YY_given_X = S_YY - S_YX @ np.linalg.solve(S_XX, S_XY)

            sign_full, log_det_full = np.linalg.slogdet(S_YY.real)
            sign_cond, log_det_cond = np.linalg.slogdet(S_YY_given_X.real)

            gc_val = log_det_full - log_det_cond
            GC[fi] = max(gc_val, 0.0)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
        except np.linalg.LinAlgError:
            GC[fi] = 0.0

    return GC


def compute_phase_lag(
    S_Z: np.ndarray,
    x_idx: list[int],
    y_idx: list[int],
    freqs: np.ndarray,
) -> np.ndarray:
    """Phase lead-lag in months at each frequency."""
    n_freqs = len(freqs)
    lag = np.zeros(n_freqs)
    for fi in range(n_freqs):
        S_XY = S_Z[fi, x_idx[0], y_idx[0]]
        phase = np.angle(S_XY)
        freq = freqs[fi]
        if freq > 0:
            lag[fi] = phase / (2.0 * np.pi * freq)
    return lag


def compute_partial_spectral_coherence(
    S_Z: np.ndarray,
    x_idx: list[int],
    y_idx: list[int],
    freqs: np.ndarray,
) -> np.ndarray:
    """Partial spectral coherence between X and Y, controlling for rest."""
    n_freqs = len(freqs)
    PSC = np.zeros(n_freqs)
    for fi in range(n_freqs):
        try:
            S_inv = np.linalg.inv(S_Z[fi])
            xi, yi = x_idx[0], y_idx[0]
            PSC[fi] = (
                np.abs(S_inv[xi, yi]) ** 2
                / (S_inv[xi, xi].real * S_inv[yi, yi].real + 1e-12)
            )
        except np.linalg.LinAlgError:
            PSC[fi] = 0.0
    return PSC


def run_causality_analysis() -> None:
    """Phase C1 entry point."""
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    out_dir = Path(config.OUTPUTS_DIR) / "phase_C"
    out_dir.mkdir(parents=True, exist_ok=True)

    Z_hat = pd.read_csv(proc_dir / "Z_hat.csv", index_col=0, parse_dates=True)
    with open(proc_dir / "split_indices.json") as fh:
        split = json.load(fh)

    Z_train = Z_hat.values[: split["train_end"]]
    logger.info("Z_hat: %s, Z_train: %s", Z_hat.shape, Z_train.shape)

    var_fit = fit_var_model(Z_train, config.VAR_MAX_LAG)

<<<<<<< HEAD
    try:
        aic_val = float(var_fit.aic)
        bic_val = float(var_fit.bic)
    except Exception:
        aic_val = 0.0
        bic_val = 0.0

    var_info = {
        "selected_lag": int(var_fit.k_ar),
        "aic": aic_val,
        "bic": bic_val,
=======
    var_info = {
        "selected_lag": int(var_fit.k_ar),
        "aic": float(var_fit.aic),
        "bic": float(var_fit.bic),
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
        "d": int(Z_hat.shape[1]),
    }
    with open(out_dir / "var_lag_order.json", "w") as fh:
        json.dump(var_info, fh, indent=2)

    n_freqs = 500
    S_Z = compute_var_spectral_density(var_fit, n_freqs)
    freqs_500 = np.linspace(0, np.pi, n_freqs)

    d_vars = config.DRIVER_VARS
    c_vars = config.CLIMATE_VARS
    rcl_vars = config.CLIMATE_RISK_VARS
    rfin_vars = config.FINANCIAL_RISK_VARS
    all_vars = list(Z_hat.columns)

    def var_indices(names: list[str]) -> list[int]:
        return [all_vars.index(n) for n in names if n in all_vars]

    d_idx = var_indices(d_vars)
    c_idx = var_indices(c_vars)
    rcl_idx = var_indices(rcl_vars)
    rfin_idx = var_indices(rfin_vars)

    pairs = [
        ("D->C", d_idx, c_idx),
        ("D->R_cl", d_idx, rcl_idx),
        ("D->R_fin", d_idx, rfin_idx),
        ("C->R_cl", c_idx, rcl_idx),
        ("C->R_fin", c_idx, rfin_idx),
        ("R_cl->R_fin", rcl_idx, rfin_idx),
    ]

    pair_names = [p[0] for p in pairs]
    GC_all = np.zeros((len(pairs), n_freqs))
    total_gc_arr = np.zeros(len(pairs))
    summary_rows: list[dict] = []

    for pi, (name, x_idx, y_idx) in enumerate(pairs):
        if not x_idx or not y_idx:
            logger.warning("Skipping %s: empty index set", name)
            continue
        GC = geweke_spectral_gc(S_Z, x_idx, y_idx, freqs_500)
        GC_all[pi] = GC
        total_gc = float(np.mean(GC) * np.pi)
        total_gc_arr[pi] = total_gc
        peak_idx = int(np.argmax(GC))
        peak_freq = float(freqs_500[peak_idx])
        peak_gc = float(GC[peak_idx])
        period = 1.0 / peak_freq if peak_freq > 0 else np.inf
        summary_rows.append({
            "pair": name,
            "total_gc": round(total_gc, 6),
            "peak_frequency_hz": round(peak_freq, 6),
            "peak_gc_value": round(peak_gc, 6),
            "dominant_period_months": round(period, 2),
        })
        logger.info("  %s: total_gc=%.4f, peak_period=%.1f months", name, total_gc, period)

    PSC = compute_partial_spectral_coherence(S_Z, rcl_idx, rfin_idx, freqs_500)

    pd.DataFrame(summary_rows).to_csv(out_dir / "causality_summary.csv", index=False)

    save_npz(
        str(out_dir / "causality_results.npz"),
        GC_all=GC_all,
        freqs=freqs_500,
        PSC=PSC,
        total_gc=total_gc_arr,
    )

    with open(out_dir / "pair_names.json", "w") as fh:
        json.dump(pair_names, fh)

<<<<<<< HEAD
    harm_fam = detect_harmonic_families(S_Z, freqs_500)
    with open(out_dir / "harmonic_families.json", "w") as fh:
        json.dump(harm_fam, fh, indent=2)

=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    logger.info("Phase C complete: %d causal pairs analysed.", len(pairs))
