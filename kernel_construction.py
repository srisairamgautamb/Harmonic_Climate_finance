"""
Phase F1 -- Harmonic Kernel Construction

Modified Green's function kernel, Laplace temporal kernel, geodesic
distances, quantum geometric kernel, and final HQG product kernel.

Incorporates Moderate Issue 7 fix: quantum geometric kernel uses a
Matern-3/2 kernel on Fisher-Rao geodesic distances instead of the
cosine-based approximation that can destroy positive definiteness.
"""

import logging
from pathlib import Path

import numpy as np

import config
from utils import ensure_positive_definite, log_shape, save_npz

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)


def compute_K_spatial(
    V_basis: np.ndarray,
    eigenvalues: np.ndarray,
    c: float,
) -> np.ndarray:
    """Modified Green's function kernel G_{M,c}.

    G_{M,c}(theta_i, theta_j) = V @ diag(1/(eigenvalues + c)) @ V.T
    """
    lambda_tilde = eigenvalues + c
    safe_lambda = np.where(np.abs(lambda_tilde) > 1e-12, lambda_tilde, 1e-12)
    K = V_basis @ np.diag(1.0 / safe_lambda) @ V_basis.T
    K = ensure_positive_definite(K, name="K_spatial")
<<<<<<< HEAD
    
    # Spherical Normalisation to bound values strictly [-1, 1] with diagonal 1.0
    d = np.diag(K)
    d = np.where(d > 1e-12, d, 1e-12)
    K = K / np.sqrt(np.outer(d, d))
    
=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    log_shape(K, "K_spatial")
    return K


def compute_kappa_alpha(
    t_centers: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Laplace temporal kernel kappa_alpha(t_i, t_j) = exp(-alpha |t_i - t_j|)."""
    n = len(t_centers)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = np.exp(-alpha * abs(t_centers[i] - t_centers[j]))
            K[j, i] = K[i, j]
    log_shape(K, "kappa_alpha")
    return K


def compute_geodesic_distances(
    theta_hat: np.ndarray,
    g_metric: np.ndarray,
) -> np.ndarray:
    """Approximate Fisher-Rao geodesic distance between all pairs.

    d_FR(i,j)^2 = (theta_i - theta_j)^T @ g_mid @ (theta_i - theta_j)
    """
    n = theta_hat.shape[0]
    d_FR = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = theta_hat[i] - theta_hat[j]
            g_mid = (g_metric[i] + g_metric[j]) / 2.0
            g_mid = (g_mid + g_mid.T) / 2.0
            try:
                dist_sq = float(diff @ g_mid @ diff)
                d_FR[i, j] = np.sqrt(max(dist_sq, 0.0))
                d_FR[j, i] = d_FR[i, j]
            except Exception:
                d_FR[i, j] = np.linalg.norm(diff)
                d_FR[j, i] = d_FR[i, j]
    log_shape(d_FR, "d_FR")
    return d_FR


def _matern32_kernel(
    d: np.ndarray,
    lengthscale: float,
) -> np.ndarray:
    """Matern-3/2 kernel: (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)."""
    r = np.sqrt(3.0) * d / max(lengthscale, 1e-8)
    return (1.0 + r) * np.exp(-r)


def compute_K_QG(
    theta_hat: np.ndarray,
    d_FR: np.ndarray,
    dominant_freqs: np.ndarray,
    alpha_QG: float,
) -> np.ndarray:
    """Quantum geometric kernel using Matern-3/2 on geodesic distances.

<<<<<<< HEAD
    Applies the Median Heuristic to prevent exponential collapse on massive manifolds.
    Guarantees positive definiteness by Schoenberg's theorem.
    """
    # Median Heuristic for scale-invariant distances
    median_d = np.median(d_FR)
    if median_d > 1e-6:
        dynamic_lengthscale = median_d
    else:
        dynamic_lengthscale = alpha_QG

    logger.info("Matern lengthscale (Median Heuristic): %.4e", dynamic_lengthscale)

    K_QG = _matern32_kernel(d_FR, lengthscale=dynamic_lengthscale)
=======
    Guarantees positive definiteness by Schoenberg's theorem.
    """
    K_QG = _matern32_kernel(d_FR, lengthscale=alpha_QG)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    K_QG = ensure_positive_definite(K_QG, name="K_QG")
    log_shape(K_QG, "K_QG")
    return K_QG


def run_kernel_construction() -> None:
<<<<<<< HEAD
    """Phase F1 entry point.
    
    Architecture fix: Use a convex combination kernel rather than Hadamard product.
    K_HQG = lam * K_QG_full + (1 - lam) * K_Harm
    
    This preserves the discriminative quantum signal which the Hadamard product
    was destroying by multiplying it by a near-degenerate K_Harm matrix.
    """
=======
    """Phase F1 entry point."""
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    out_dir = Path(config.OUTPUTS_DIR) / "phase_F"
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_data = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_E" / "Phi_coefficients.npz"),
        allow_pickle=True,
    )
    V_basis = phi_data["V_basis"]
    eigenvalues = phi_data["eigenvalues"]

    theta_data = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_D" / "theta_hat.npz"),
        allow_pickle=True,
    )
    theta_hat = theta_data["theta_hat"]

    g_data = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_D" / "fisher_rao_metric.npz"),
        allow_pickle=True,
    )
    g_metric = g_data["g_metric"]

    spectral_data = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_B" / "spectral_features.npz"),
        allow_pickle=True,
    )
    freqs = spectral_data["freqs"]
    window_centers = spectral_data["window_centers"]

    n_windows = theta_hat.shape[0]
    logger.info("n_windows=%d, theta_hat=%s", n_windows, theta_hat.shape)

<<<<<<< HEAD
    # ── Component 1: Harmonic-Temporal kernel (geometric prior) ──
=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    K_spatial = compute_K_spatial(V_basis, eigenvalues, config.C_MASS)
    K_temporal = compute_kappa_alpha(
        window_centers[:n_windows].astype(float),
        config.ALPHA_TEMPORAL,
    )[:n_windows, :n_windows]
<<<<<<< HEAD
    K_Harm = K_spatial * K_temporal
    K_Harm = ensure_positive_definite(K_Harm, name="K_Harm")
    # Normalise K_Harm to correlation scale so convex combination is balanced
    d_harm = np.diag(K_Harm)
    d_harm = np.where(d_harm > 1e-12, d_harm, 1e-12)
    K_Harm = K_Harm / np.sqrt(np.outer(d_harm, d_harm))

    # ── Component 2: Fisher-Rao geodesic Matern kernel ──
    d_FR = compute_geodesic_distances(theta_hat, g_metric)
    dominant_freqs = freqs[:config.N_QUBITS] if len(freqs) >= config.N_QUBITS else freqs
    K_QG = compute_K_QG(theta_hat, d_FR, dominant_freqs, config.ALPHA_QG)

    # ── Component 3: Quantum fidelity kernel (from Qiskit Aer simulation) ──
    # Build the full [n_windows, n_windows] quantum kernel by padding/extending train block
    qk_path = Path(config.OUTPUTS_DIR) / "phase_G" / "quantum_kernel_matrix.npz"
    K_quantum_full = None
    if qk_path.exists():
        try:
            qk_data = np.load(str(qk_path), allow_pickle=True)
            K_train = qk_data["K_train"]  # [n_train, n_train]
            K_test  = qk_data.get("K_test", None)   # [n_test, n_train] if saved
            
            import json
            from pathlib import Path as _Path
            with open(_Path(config.DATA_PROCESSED_DIR) / "split_indices.json") as _fh:
                split_info = json.load(_fh)
            
            wc = window_centers[:n_windows].astype(int)
            train_mask = wc < split_info["train_end"]
            test_mask  = wc >= split_info["train_end"]
            n_tr = train_mask.sum()
            n_te = test_mask.sum()
            
            K_quantum_full = np.zeros((n_windows, n_windows))
            # Fill train-train block
            K_quantum_full[:n_tr, :n_tr] = K_train[:n_tr, :n_tr]
            if K_test is not None and K_test.shape[0] == n_te:
                K_quantum_full[n_tr:, :n_tr] = K_test[:n_te, :n_tr]
                K_quantum_full[:n_tr, n_tr:] = K_test[:n_te, :n_tr].T
            # test-test block: use RBF approximation from theta_hat for continuity
            for i in range(n_tr, n_windows):
                for j in range(i, n_windows):
                    diff = theta_hat[i] - theta_hat[j]
                    sigma = np.median(np.linalg.norm(
                        theta_hat[n_tr:] - theta_hat[n_tr:].mean(axis=0), axis=1
                    )) + 1e-8
                    K_quantum_full[i, j] = np.exp(-np.dot(diff, diff) / (2.0 * sigma**2))
                    K_quantum_full[j, i] = K_quantum_full[i, j]
            K_quantum_full = ensure_positive_definite(K_quantum_full, name="K_quantum_full")
            # Normalise
            d_q = np.diag(K_quantum_full)
            d_q = np.where(d_q > 1e-12, d_q, 1e-12)
            K_quantum_full = K_quantum_full / np.sqrt(np.outer(d_q, d_q))
            logger.info("K_quantum_full: off-std=%.4f, off-mean=%.4f",
                        K_quantum_full[np.triu_indices(n_windows, k=1)].std(),
                        K_quantum_full[np.triu_indices(n_windows, k=1)].mean())
        except Exception as exc:
            logger.warning("Could not build K_quantum_full: %s. Using K_QG only.", exc)
            K_quantum_full = None

    # ── Convex Combination: K_HQG = lam_q*K_quantum + lam_g*K_QG + lam_h*K_Harm ──
    # Weights chosen so quantum signal dominates, geodesic provides topology,
    # harmonic-temporal acts as a smoothing regularizer.
    lam_q = 0.5   # quantum fidelity weight
    lam_g = 0.3   # geodesic Matern weight  
    lam_h = 0.2   # harmonic-temporal weight

    if K_quantum_full is not None:
        K_HQG = lam_q * K_quantum_full + lam_g * K_QG + lam_h * K_Harm
        logger.info("K_HQG built via 3-component convex combination (q=%.1f, g=%.1f, h=%.1f)",
                    lam_q, lam_g, lam_h)
    else:
        # Fallback: geodesic + harmonic
        K_HQG = 0.6 * K_QG + 0.4 * K_Harm
        logger.info("K_HQG built via 2-component fallback (no quantum kernel)")

    K_HQG = ensure_positive_definite(K_HQG, name="K_HQG")

    logger.info("K_HQG final: off-std=%.4f, off-mean=%.4f",
                K_HQG[np.triu_indices(n_windows, k=1)].std(),
                K_HQG[np.triu_indices(n_windows, k=1)].mean())

=======

    K_Harm = K_spatial * K_temporal
    K_Harm = ensure_positive_definite(K_Harm, name="K_Harm")

    d_FR = compute_geodesic_distances(theta_hat, g_metric)

    dominant_freqs = freqs[:config.N_QUBITS] if len(freqs) >= config.N_QUBITS else freqs
    K_QG = compute_K_QG(theta_hat, d_FR, dominant_freqs, config.ALPHA_QG)

    K_HQG = K_Harm * K_QG
    K_HQG = ensure_positive_definite(K_HQG, name="K_HQG")

>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    save_npz(str(out_dir / "K_Harm.npz"), K=K_Harm, d_FR=d_FR)
    save_npz(str(out_dir / "K_QG.npz"), K=K_QG)
    save_npz(str(out_dir / "K_HQG.npz"), K=K_HQG)

    logger.info("Phase F complete: K_Harm, K_QG, K_HQG saved.")
<<<<<<< HEAD

=======
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
