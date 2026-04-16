"""
Unit tests for the Spectral Climate-Financial Risk Transmission pipeline.

Run with:  pytest tests/test_pipeline.py -v --tb=short
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_config_dimensions():
    """Config variable lists have the correct counts."""
    from config import CLIMATE_RISK_VARS, CLIMATE_VARS, DRIVER_VARS, FINANCIAL_RISK_VARS

    assert len(DRIVER_VARS) == 9
    assert len(CLIMATE_VARS) == 6
    assert len(CLIMATE_RISK_VARS) == 4
    assert len(FINANCIAL_RISK_VARS) == 6


def test_stationarity_test():
    """Stationarity function returns valid ADF/KPSS p-values."""
    np.random.seed(42)
    from preprocessing import run_stationarity_tests

    s = pd.Series(np.random.randn(200))
    result = run_stationarity_tests(s, "test_var")
    assert "adf_pval" in result
    assert "kpss_pval" in result
    assert 0.0 <= result["adf_pval"] <= 1.0


def test_dft_shape():
    """DFT output has the expected number of frequency bins."""
    np.random.seed(42)
    from config import N_FREQ, WINDOW_SIZE
    from spectral_estimation import compute_windowed_dft

    x = np.random.randn(WINDOW_SIZE)
    freqs, amps, phases = compute_windowed_dft(x, WINDOW_SIZE, "hann")
    assert len(freqs) == N_FREQ
    assert len(amps) == N_FREQ
    assert len(phases) == N_FREQ


def test_coherence_bounds():
    """Coherence values lie in [0, 1]."""
    np.random.seed(42)
    from config import N_FREQ
    from spectral_estimation import compute_coherence

    d_U = 5
    S_UU = np.random.randn(d_U, d_U, N_FREQ) + 1j * np.random.randn(d_U, d_U, N_FREQ)
    for f in range(N_FREQ):
        S_UU[:, :, f] = S_UU[:, :, f] @ S_UU[:, :, f].conj().T + 5 * np.eye(d_U)
    Coh = compute_coherence(S_UU)
    assert Coh.min() >= -1e-8
    assert Coh.max() <= 1.0 + 1e-8


def test_harmonic_kernel_pd():
    """Harmonic kernel is positive definite after jitter."""
    np.random.seed(42)
    from kernel_construction import compute_K_spatial, compute_kappa_alpha
    from utils import ensure_positive_definite

    n = 20
    V = np.random.randn(n, 10)
    V, _ = np.linalg.qr(V)
    eigs = np.abs(np.random.randn(10)) + 0.1
    K_spat = compute_K_spatial(V, eigs, c=1.0)
    t_centers = np.arange(n, dtype=float)
    K_temp = compute_kappa_alpha(t_centers, alpha=0.1)
    K_Harm = K_spat * K_temp
    K_Harm = ensure_positive_definite(K_Harm, name="K_Harm_test")
    eigvals = np.linalg.eigvalsh(K_Harm)
    assert eigvals.min() > -1e-8


def test_transmission_jacobian_shape():
    """Jacobian of a linear function has the expected shape."""
    import torch

    q_fin, k_param = 6, 5
    theta = torch.randn(k_param, requires_grad=True)
    W = torch.randn(q_fin, k_param)
    J = torch.autograd.functional.jacobian(lambda t: W @ t, theta)
    assert J.shape == (q_fin, k_param)


def test_whittle_likelihood():
    """Whittle likelihood returns a finite scalar."""
    import torch

    from config import N_FREQ
    from var_spectral_param import whittle_log_likelihood, var_to_spectral

    d_U, p_0 = 3, 2
    n_off = d_U * (d_U - 1) // 2
    k = d_U ** 2 * p_0 + n_off + d_U
    theta = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    freqs = torch.linspace(0.01, 0.49, N_FREQ, dtype=torch.float64)
    I_fake = torch.eye(d_U, dtype=torch.complex128).unsqueeze(0).repeat(N_FREQ, 1, 1)
    ll = whittle_log_likelihood(theta, I_fake, freqs)
    assert torch.isfinite(ll), f"Whittle likelihood is not finite: {ll}"


def test_fisher_rao_metric():
    """Fisher-Rao metric is symmetric and PSD."""
    np.random.seed(42)
    from config import N_FREQ
    from var_spectral_param import compute_fisher_rao_metric

    d_U, p_0 = 3, 2
    n_off = d_U * (d_U - 1) // 2
    k = d_U ** 2 * p_0 + n_off + d_U
    theta = np.zeros(k)
    freqs = np.linspace(0.01, 0.49, N_FREQ)
    g = compute_fisher_rao_metric(theta, freqs, d_U, p_0)
    assert g.shape == (k, k)
    assert np.allclose(g, g.T, atol=1e-6)
    eigvals = np.linalg.eigvalsh(g)
    assert eigvals.min() >= -1e-8


def test_dm_test():
    """Diebold-Mariano test returns a valid p-value."""
    np.random.seed(42)
    from utils import diebold_mariano_test

    e1 = np.random.randn(100)
    e2 = np.random.randn(100) * 2
    result = diebold_mariano_test(e1, e2)
    assert 0.0 <= result["p_value"] <= 1.0
    assert "DM_stat" in result
    assert "reject_H0_at_5pct" in result


def test_no_nan_after_alignment():
    """Aligned monthly data has no NaN values."""
    import os

    path = "data/processed/aligned_monthly.csv"
    if not os.path.exists(path):
        pytest.skip("Aligned data not yet generated")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    nan_sums = df.isnull().sum()
    assert nan_sums.sum() == 0, f"NaN values found: {nan_sums[nan_sums > 0]}"
