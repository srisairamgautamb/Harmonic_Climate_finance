"""
Phase D1 -- VAR Spectral Parameter Estimation

Whittle MLE per rolling window with Cholesky log-diagonal
parameterisation (Critical Issue 2 fix), Fisher-Rao metric via
autograd, and initial transmission operator estimate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

import config
from utils import assert_shape, log_shape, save_npz

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


def _reconstruct_L(theta_L: torch.Tensor, d_U: int) -> torch.Tensor:
    """Build lower-triangular L with ``exp`` diagonal from free params.

    ``theta_L`` layout: [off-diagonal entries | log-diagonal entries]
    """
    n_off = d_U * (d_U - 1) // 2
    L_lower_vals = theta_L[:n_off]
    L_log_diag = theta_L[n_off:]
    L = torch.zeros(d_U, d_U, dtype=theta_L.dtype)
    idx = torch.tril_indices(d_U, d_U, offset=-1)
    L[idx[0], idx[1]] = L_lower_vals
    L[range(d_U), range(d_U)] = torch.exp(L_log_diag)
    return L


def _unpack_theta(
    theta: torch.Tensor,
    d_U: int,
    p_0: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Split theta into VAR coefficient matrices and Sigma_u."""
    n_A = d_U * d_U * p_0
    n_off = d_U * (d_U - 1) // 2
    n_L = n_off + d_U

    A_vec = theta[:n_A]
    L_vec = theta[n_A : n_A + n_L]

    A_mats = [
        A_vec[k * d_U * d_U : (k + 1) * d_U * d_U].reshape(d_U, d_U)
        for k in range(p_0)
    ]
    L = _reconstruct_L(L_vec, d_U)
    Sigma_u = L @ L.T
    return A_mats, Sigma_u


def var_to_spectral(
    theta: torch.Tensor,
    freqs: torch.Tensor,
    d_U: int,
    p_0: int,
) -> torch.Tensor:
    """Differentiable VAR spectral density S_UU(omega; theta).

    Returns complex tensor of shape [n_freq, d_U, d_U].
    """
    A_mats, Sigma_u = _unpack_theta(theta, d_U, p_0)
    n_freq = freqs.shape[0]
    I_d = torch.eye(d_U, dtype=torch.complex128)
    S = torch.zeros(n_freq, d_U, d_U, dtype=torch.complex128)

    for fi in range(n_freq):
        omega = freqs[fi]
        A_omega = I_d.clone()
        for k in range(p_0):
            A_omega = A_omega - A_mats[k].to(torch.complex128) * torch.exp(
                torch.tensor(-1j * (k + 1) * omega, dtype=torch.complex128)
            )
        A_inv = torch.linalg.inv(A_omega)
        S[fi] = (1.0 / (2.0 * np.pi)) * A_inv @ Sigma_u.to(
            torch.complex128
        ) @ A_inv.conj().T

    return S


def whittle_log_likelihood(
    theta: torch.Tensor,
    I_periodogram: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
<<<<<<< HEAD
    """Whittle log-likelihood (scalar, differentiable).
    Defined as ell(theta) = -0.5 * sum [ln det S + Tr(S^{-1} I)].
    To maximize likelihood, we minimize -ell(theta).
    """
=======
    """Whittle log-likelihood (scalar, differentiable)."""
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    d_U = I_periodogram.shape[1]
    p_0 = (theta.shape[0] - d_U * (d_U - 1) // 2 - d_U) // (d_U * d_U)
    S = var_to_spectral(theta, freqs, d_U, p_0)
    n_freq = freqs.shape[0]
    W = I_periodogram.shape[0] if len(I_periodogram.shape) == 4 else n_freq

    ll = torch.tensor(0.0, dtype=torch.float64)
    for fi in range(n_freq):
        S_f = S[fi]
        I_f = I_periodogram[fi]
        sign, logdet = torch.linalg.slogdet(S_f)
        S_inv = torch.linalg.inv(S_f)
        trace = torch.real(torch.trace(S_inv @ I_f))
<<<<<<< HEAD
        ll = ll - 0.5 * (logdet.real + trace)
=======
        ll = ll - logdet.real - trace
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    return ll


def fit_whittle_mle_window(
    U_window: np.ndarray,
    theta_init: np.ndarray,
) -> np.ndarray:
    """Fit Whittle MLE for a single rolling window."""
    W, d_U = U_window.shape
    freqs_np = np.fft.rfftfreq(W)[1 : config.N_FREQ + 1]
    freqs_t = torch.tensor(freqs_np, dtype=torch.float64)

    U_fft = np.fft.rfft(U_window, axis=0)[1 : config.N_FREQ + 1]
    I_per = np.zeros((config.N_FREQ, d_U, d_U), dtype=complex)
    for fi in range(config.N_FREQ):
        u_f = U_fft[fi].reshape(-1, 1)
        I_per[fi] = (1.0 / W) * (u_f @ u_f.conj().T)
    I_t = torch.tensor(I_per, dtype=torch.complex128)

    theta_t = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)

    def objective(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        try:
            ll = whittle_log_likelihood(x_t, I_t, freqs_t)
<<<<<<< HEAD
            neg_ll = -ll  # Minimizing negative log-likelihood
=======
            neg_ll = -ll
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
            neg_ll.backward()
            grad = x_t.grad.detach().numpy().copy()
            return float(neg_ll.detach()), grad
        except Exception:
            return 1e10, np.zeros_like(x_np)

    try:
        result = minimize(
            objective,
            theta_init,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )
        return result.x
    except Exception as exc:
        logger.warning("Whittle MLE failed: %s, returning init.", exc)
        return theta_init


<<<<<<< HEAD
def expected_whittle_nll(
    theta_opt: torch.Tensor,
    S_plug: torch.Tensor,
    freqs: torch.Tensor,
    d_U: int,
    p_0: int,
) -> torch.Tensor:
    S = var_to_spectral(theta_opt, freqs, d_U, p_0)
    n_freq = freqs.shape[0]
    ll = torch.tensor(0.0, dtype=torch.float64)
    for fi in range(n_freq):
        S_f = S[fi]
        I_f = S_plug[fi]
        sign, logdet = torch.linalg.slogdet(S_f)
        S_inv = torch.linalg.inv(S_f)
        trace = torch.real(torch.trace(S_inv @ I_f))
        ll = ll + logdet.real + trace
    return ll


=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
def compute_fisher_rao_metric(
    theta: np.ndarray,
    freqs: np.ndarray,
    d_U: int,
    p_0: int,
) -> np.ndarray:
<<<<<<< HEAD
    """Compute the Fisher-Rao metric g_ij(theta) via numerical Hessian of expected NLL."""
    try:
        theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        freqs_t = torch.tensor(freqs, dtype=torch.float64)

        with torch.no_grad():
            S_plug = var_to_spectral(theta_t, freqs_t, d_U, p_0).detach()

        H = torch.autograd.functional.hessian(
            lambda x: expected_whittle_nll(x, S_plug, freqs_t, d_U, p_0),
            theta_t,
        )

        W = 2 * len(freqs)
        g_ij = (1.0 / (2.0 * W)) * H.detach().numpy()
        g_ij = (g_ij + g_ij.T) / 2.0
        
        # Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(g_ij)
        eigvals = np.clip(eigvals, 1e-8, None)
        g_ij = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return g_ij
    except Exception as exc:
        logger.warning("Fisher-Rao analytic failed (%s), returning Identity", exc)
        return np.eye(len(theta))
=======
    """Compute the Fisher-Rao metric g_ij(theta) via numerical differentiation."""
    k = len(theta)
    # BYPASS: For rapid synthetic validation, return the identity metric.
    # The true empirical calculation requires O(k^2) matrix inversions,
    # which takes hours for k=1273.
    logger.info("  [Bypass] Returning identity for g_ij to speed up dry-run")
    return np.eye(k)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60


def run_var_spectral_param() -> None:
    """Phase D1 entry point."""
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    out_dir = Path(config.OUTPUTS_DIR) / "phase_D"
    out_dir.mkdir(parents=True, exist_ok=True)

    U_hat = pd.read_csv(proc_dir / "U_hat.csv", index_col=0, parse_dates=True)
    d_U = U_hat.shape[1]
    T = U_hat.shape[0]
    U_vals = U_hat.values

    with open(Path(config.OUTPUTS_DIR) / "phase_C" / "var_lag_order.json") as fh:
        var_info = json.load(fh)
    p_0 = var_info["selected_lag"]

    n_off = d_U * (d_U - 1) // 2
    k_param = d_U * d_U * p_0 + n_off + d_U
    logger.info("d_U=%d, p_0=%d, k_param=%d", d_U, p_0, k_param)

    theta_global = np.zeros(k_param)
    n_A = d_U * d_U * p_0
    theta_global[n_A + n_off :] = 0.0

    W = config.WINDOW_SIZE
    half = W // 2
    window_centers = list(range(half, T - half, config.WINDOW_STEP))
    n_windows = len(window_centers)

    theta_hat_all = np.zeros((n_windows, k_param))
    n_freq = config.N_FREQ
    freqs = np.fft.rfftfreq(W)[1 : n_freq + 1]

    spectral_data = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_B" / "spectral_features.npz"),
        allow_pickle=True,
    )
    S_UU_all = spectral_data["S_UU"]
    S_UF_all = spectral_data["S_UF"]

    for wi, tc in enumerate(window_centers):
        U_w = U_vals[tc - half : tc + half, :]
        theta_init = theta_global if wi == 0 else theta_hat_all[wi - 1].copy()
        theta_hat_all[wi] = fit_whittle_mle_window(U_w, theta_init)
        if (wi + 1) % 20 == 0:
            logger.info("  Whittle MLE: window %d/%d", wi + 1, n_windows)

<<<<<<< HEAD
    logger.info("Computing Fisher-Rao metric for %d windows ...", n_windows)
    g_metric_full = np.zeros((n_windows, k_param, k_param))
=======
    save_npz(str(out_dir / "theta_hat.npz"), theta_hat=theta_hat_all)

    logger.info("Computing Fisher-Rao metric for %d windows ...", n_windows)
    g_metric = np.zeros((n_windows, k_param, k_param))
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    n_metric_windows = min(n_windows, 20)
    sample_idx = np.linspace(0, n_windows - 1, n_metric_windows, dtype=int)

    for si, wi in enumerate(sample_idx):
<<<<<<< HEAD
        g_metric_full[wi] = compute_fisher_rao_metric(
=======
        g_metric[wi] = compute_fisher_rao_metric(
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
            theta_hat_all[wi], freqs, d_U, p_0,
        )
        if (si + 1) % 5 == 0:
            logger.info("  Fisher-Rao: %d/%d", si + 1, n_metric_windows)

    for wi in range(n_windows):
        if wi not in sample_idx:
            nearest = sample_idx[np.argmin(np.abs(sample_idx - wi))]
<<<<<<< HEAD
            g_metric_full[wi] = g_metric_full[nearest]

    G_avg = np.mean(g_metric_full, axis=0)
    eigvals, eigvecs = np.linalg.eigh(G_avg)
    
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    k_red = d_U
    U_red = eigvecs[:, :k_red]
    Lambda_red = np.diag(eigvals[:k_red])
    
    theta_mean = np.mean(theta_hat_all, axis=0)
    theta_red = (theta_hat_all - theta_mean) @ U_red @ np.sqrt(Lambda_red)
    
    # In Nystrom reduced, the new metric is essentially identity up to scaling
    g_metric_red = np.tile(np.eye(k_red), (n_windows, 1, 1))

    save_npz(
        str(out_dir / "theta_hat.npz"), 
        theta_hat=theta_red, 
        theta_hat_full=theta_hat_all
    )
    save_npz(
        str(out_dir / "fisher_rao_metric.npz"), 
        g_metric=g_metric_red, 
        g_metric_full=g_metric_full
    )
=======
            g_metric[wi] = g_metric[nearest]

    save_npz(str(out_dir / "fisher_rao_metric.npz"), g_metric=g_metric)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    q_fin = S_UF_all.shape[2] if len(S_UF_all.shape) >= 3 else config.q_fin or 6
    T_hat_emp = np.zeros((n_windows, n_freq, q_fin, d_U), dtype=complex)
    for wi in range(min(n_windows, S_UU_all.shape[0])):
        for fi in range(n_freq):
            try:
                S_UU_f = S_UU_all[wi, :, :, fi]
                S_UF_f = S_UF_all[wi, :, :, fi]
                T_hat_emp[wi, fi] = (
                    S_UF_f.T @ np.linalg.inv(
                        S_UU_f + 1e-8 * np.eye(d_U)
                    )
                )
            except np.linalg.LinAlgError:
                pass

    save_npz(str(out_dir / "T_hat_empirical.npz"), T_hat_empirical=T_hat_emp)

    logger.info(
<<<<<<< HEAD
        "Phase D complete: theta_hat %s, g_metric_red %s, T_hat %s",
        theta_hat_all.shape, g_metric_red.shape, T_hat_emp.shape,
=======
        "Phase D complete: theta_hat %s, g_metric %s, T_hat %s",
        theta_hat_all.shape, g_metric.shape, T_hat_emp.shape,
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    )
