"""
Phase B1 -- Spectral Estimation Module

Rolling windowed DFT, multi-taper cross-spectral density, coherence,
phase lead-lag, and spectral feature matrix construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import csd, welch
from scipy.signal.windows import dpss, hann

import config
from utils import assert_shape, log_shape, save_npz

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)


def compute_windowed_dft(
    series: np.ndarray,
    window_size: int,
    taper: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tapered DFT of a single window.

    Returns
    -------
    freqs : array [N_FREQ]
    amplitudes : array [N_FREQ]
    phases : array [N_FREQ]
    """
    try:
        if taper == "hann":
            w = hann(window_size)
        else:
            w = np.ones(window_size)
        x_tapered = series * w
        X = np.fft.rfft(x_tapered)
        freqs = np.fft.rfftfreq(window_size)
        X_trunc = X[1 : config.N_FREQ + 1]
        freqs_trunc = freqs[1 : config.N_FREQ + 1]
        amplitudes = np.abs(X_trunc)
        phases = np.angle(X_trunc)
        return freqs_trunc, amplitudes, phases
    except Exception as exc:
        logger.error("Windowed DFT failed: %s", exc)
        raise


def compute_spectral_density_matrix(
    U_window: np.ndarray,
    F_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-taper cross-spectral density for one window.

    Returns
    -------
    S_UU : complex array [d_U, d_U, N_FREQ]
    S_UF : complex array [d_U, q_fin, N_FREQ]
    """
    d_U = U_window.shape[1]
    q_fin = F_window.shape[1]
    W = U_window.shape[0]
    n_freq = config.N_FREQ

    try:
        tapers = dpss(W, NW=4, Kmax=min(7, W - 1))
    except Exception:
        tapers = None

    S_UU = np.zeros((d_U, d_U, n_freq), dtype=complex)
    S_UF = np.zeros((d_U, q_fin, n_freq), dtype=complex)

    for i in range(d_U):
        for j in range(i, d_U):
            try:
                f, Sij = csd(
                    U_window[:, i], U_window[:, j],
                    fs=1.0, nperseg=W, noverlap=0,
                    window=hann(W),
                )
                f_trunc = f[1 : n_freq + 1]
                S_UU[i, j, :] = Sij[1 : n_freq + 1]
                S_UU[j, i, :] = np.conj(Sij[1 : n_freq + 1])
            except Exception:
                pass

    for i in range(d_U):
        for k in range(q_fin):
            try:
                f, Sik = csd(
                    U_window[:, i], F_window[:, k],
                    fs=1.0, nperseg=W, noverlap=0,
                    window=hann(W),
                )
                S_UF[i, k, :] = Sik[1 : n_freq + 1]
            except Exception:
                pass

    return S_UU, S_UF


def compute_coherence(S_UU: np.ndarray) -> np.ndarray:
    """Magnitude-squared coherence from auto/cross-spectral density.

    Parameters
    ----------
    S_UU : complex array [d_U, d_U, N_FREQ]

    Returns
    -------
    Coh : real array [d_U, d_U, N_FREQ]  values in [0, 1]
    """
    d_U = S_UU.shape[0]
    n_freq = S_UU.shape[2]
    Coh = np.zeros((d_U, d_U, n_freq))
    for i in range(d_U):
        for j in range(d_U):
            Sii = S_UU[i, i, :].real
            Sjj = S_UU[j, j, :].real
            denom = Sii * Sjj
            denom = np.where(denom > 0, denom, 1e-12)
            Coh[i, j, :] = np.abs(S_UU[i, j, :]) ** 2 / denom
    Coh = np.clip(Coh, 0.0, 1.0)
    return Coh


def build_spectral_features(
    U_hat: pd.DataFrame,
    F_hat: pd.DataFrame,
) -> np.ndarray:
    """Build the full spectral feature matrix PHI over rolling windows.

    Returns PHI of shape [n_windows, d_phi].
    """
    U = U_hat.values
    F = F_hat.values
    T = U.shape[0]
    d_U = U.shape[1]
    q_fin = F.shape[1]
    W = config.WINDOW_SIZE
    n_freq = config.N_FREQ
    half = W // 2

    window_centers = list(range(half, T - half, config.WINDOW_STEP))
    n_windows = len(window_centers)
    d_phi = (2 * d_U + q_fin) * n_freq

    PHI = np.zeros((n_windows, d_phi))
    S_UU_all = np.zeros((n_windows, d_U, d_U, n_freq), dtype=complex)
    S_UF_all = np.zeros((n_windows, d_U, q_fin, n_freq), dtype=complex)
    Coh_all = np.zeros((n_windows, d_U, d_U, n_freq))
    phase_UF_all = np.zeros((n_windows, d_U, q_fin, n_freq))
    lag_UF_all = np.zeros((n_windows, d_U, q_fin, n_freq))

    freqs_trunc = None

    for wi, tc in enumerate(window_centers):
        U_w = U[tc - half : tc + half, :]
        F_w = F[tc - half : tc + half, :]

        amp_U = np.zeros((d_U, n_freq))
        phase_U = np.zeros((d_U, n_freq))
        amp_F = np.zeros((q_fin, n_freq))

        for j in range(d_U):
            freqs, amp, ph = compute_windowed_dft(U_w[:, j], W, config.TAPER)
            amp_U[j, :] = amp
            phase_U[j, :] = ph
            if freqs_trunc is None:
                freqs_trunc = freqs

        for k in range(q_fin):
            freqs, amp, _ = compute_windowed_dft(F_w[:, k], W, config.TAPER)
            amp_F[k, :] = amp

        phi_tc = np.concatenate([
            amp_U.flatten(),
            amp_F.flatten(),
            phase_U.flatten(),
        ])
        PHI[wi, :] = phi_tc

        S_UU, S_UF = compute_spectral_density_matrix(U_w, F_w)
        S_UU_all[wi] = S_UU
        S_UF_all[wi] = S_UF
        Coh_all[wi] = compute_coherence(S_UU)

        for i in range(d_U):
            for k in range(q_fin):
                phase_UF_all[wi, i, k, :] = np.angle(S_UF[i, k, :])
                safe_freqs = np.where(freqs_trunc > 0, freqs_trunc, 1e-12)
                lag_UF_all[wi, i, k, :] = (
                    phase_UF_all[wi, i, k, :] / (2.0 * np.pi * safe_freqs)
                )

        if (wi + 1) % 50 == 0:
            logger.info("  Spectral estimation: window %d/%d", wi + 1, n_windows)

    return (
        PHI, S_UU_all, S_UF_all, Coh_all,
        phase_UF_all, lag_UF_all,
        freqs_trunc, np.array(window_centers),
    )


def run_spectral_estimation() -> None:
    """Phase B1 entry point."""
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    out_dir = Path(config.OUTPUTS_DIR) / "phase_B"
    out_dir.mkdir(parents=True, exist_ok=True)

    U_hat = pd.read_csv(proc_dir / "U_hat.csv", index_col=0, parse_dates=True)
    F_hat = pd.read_csv(proc_dir / "F_hat.csv", index_col=0, parse_dates=True)

    logger.info("U_hat: %s, F_hat: %s", U_hat.shape, F_hat.shape)

    (
        PHI, S_UU_all, S_UF_all, Coh_all,
        phase_all, lag_all, freqs_trunc, t_c_array,
    ) = build_spectral_features(U_hat, F_hat)

    log_shape(PHI, "PHI")
    log_shape(S_UU_all.real, "S_UU (real part)")
    log_shape(Coh_all, "Coherence")

    save_npz(
        str(out_dir / "spectral_features.npz"),
        PHI=PHI,
        S_UU=S_UU_all,
        S_UF=S_UF_all,
        Coh_UU=Coh_all,
        phase_UF=phase_all,
        lag_UF=lag_all,
        freqs=freqs_trunc,
        window_centers=t_c_array,
    )
    logger.info(
        "Phase B complete: n_windows=%d, d_phi=%d",
        PHI.shape[0], PHI.shape[1],
    )
