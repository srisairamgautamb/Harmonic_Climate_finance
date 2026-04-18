"""
Phase Topological -- Topology & Classification

Constructs spectral networks using Coh_UU, evaluates Vietoris-Rips
persistence using gudhi, builds K_PD, and clusters phases using Gap Stat KMeans.
"""

import json
import logging
from pathlib import Path

import numpy as np
import gudhi
from gudhi.wasserstein import wasserstein_distance
try:
    from gap_statistic import OptimalK
except ImportError:
    OptimalK = None
from sklearn.cluster import KMeans

import config
from utils import save_npz

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)


def compute_wasserstein_kernel(pd_list: list[list[tuple]], sigma: float = 1.0) -> np.ndarray:
    """Compute K_PD from a list of diagrams (list of tuples (birth, death))."""
    n = len(pd_list)
    d_mat = np.zeros((n, n))
    
    # Form properly shaped arrays for wasserstein
    clean_pds = []
    for diag in pd_list:
        if len(diag) == 0:
            clean_pds.append(np.empty((0, 2)))
        else:
            clean_pds.append(np.array(diag))

    for i in range(n):
        for j in range(i, n):
            d_w = wasserstein_distance(clean_pds[i], clean_pds[j], order=2, internal_p=2)
            d_mat[i, j] = d_w
            d_mat[j, i] = d_w
            
    K_PD = np.exp(- (d_mat ** 2) / (sigma ** 2))
    return K_PD


def _build_climate_finance_distance(S_UU: np.ndarray, S_UF: np.ndarray) -> np.ndarray:
    """Build a joint climate–finance spectral distance matrix.

    Constructs a (d_U + q_fin) × (d_U + q_fin) distance matrix from the
    cross-spectral block. Nodes = variables, edge weight = 1 − mean |coherence|.
    This produces genuinely non-trivial H₁ cycles because the bipartite
    climate→finance edges create loops through variables with different
    transmission paths.
    """
    d_U, _, n_freq = S_UU.shape[0], S_UU.shape[1], S_UU.shape[2]
    q_fin = S_UF.shape[1]
    n_vars = d_U + q_fin

    dist = np.ones((n_vars, n_vars))
    np.fill_diagonal(dist, 0.0)

    # Within-upstream: use magnitude-squared coherence
    for i in range(d_U):
        for j in range(i + 1, d_U):
            denom = np.sqrt(np.abs(S_UU[i, i]) * np.abs(S_UU[j, j]) + 1e-30)
            coh = np.mean(np.abs(S_UU[i, j]) / (denom + 1e-30))
            coh = float(np.clip(coh, 0.0, 1.0))
            dist[i, j] = dist[j, i] = 1.0 - coh

    # Cross-block (upstream ↔ financial): use |S_UF| averaged over frequencies
    for i in range(d_U):
        for j in range(q_fin):
            num = float(np.mean(np.abs(S_UF[i, j])))
            s_uu_ii = float(np.mean(np.abs(S_UU[i, i])))
            s_ff_jj = float(np.mean(np.abs(S_UF[i, j]) ** 2)) + 1e-30
            denom = float(np.sqrt(s_uu_ii * s_ff_jj + 1e-30))
            coh = float(np.clip(num / (denom + 1e-30), 0.0, 1.0))
            dist[i, d_U + j] = dist[d_U + j, i] = 1.0 - coh

    # Within-financial block: Euclidean proxy via S_UF columns
    for i in range(q_fin):
        for j in range(i + 1, q_fin):
            col_i = np.abs(S_UF[:, i, :])
            col_j = np.abs(S_UF[:, j, :])
            corr = float(np.mean(col_i * col_j) /
                         (np.sqrt(np.mean(col_i**2) * np.mean(col_j**2)) + 1e-30))
            corr = np.clip(corr, 0.0, 1.0)
            dist[d_U + i, d_U + j] = dist[d_U + j, d_U + i] = 1.0 - corr

    return np.clip(dist, 0.0, 1.0)


def run_topological_analysis() -> None:
    out_dir = Path(config.OUTPUTS_DIR) / "phase_Topo"
    out_dir.mkdir(parents=True, exist_ok=True)

    spectral_data = np.load(
        str(Path(config.OUTPUTS_DIR) / "phase_B" / "spectral_features.npz"),
        allow_pickle=True,
    )
    S_UU = spectral_data["S_UU"]   # [n_windows, d_U, d_U, n_freq]
    S_UF = spectral_data["S_UF"]   # [n_windows, d_U, q_fin, n_freq]

    n_windows = S_UU.shape[0]
    pd_h1 = []
    pd_h0 = []

    logger.info("Computing Vietoris-Rips persistence on climate–finance joint network "
                "over %d windows...", n_windows)

    pers_sum_h1 = np.zeros(n_windows)

    for wi in range(n_windows):
        # Build richer joint distance using both upstream and cross-block coherence
        dist_mat = _build_climate_finance_distance(S_UU[wi], S_UF[wi])

        # Adaptive max_edge_length: use 90th percentile of off-diag distances
        # so the complex captures topology at the right scale rather than
        # connecting everything or nothing.
        off_diag = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]
        max_len = float(np.percentile(off_diag, 80))
        max_len = max(max_len, 0.05)

        rips = gudhi.RipsComplex(distance_matrix=dist_mat, max_edge_length=max_len)
        st = rips.create_simplex_tree(max_dimension=2)
        diag = st.persistence()

        h1 = [p[1] for p in diag if p[0] == 1]
        h0 = [p[1] for p in diag if p[0] == 0 and p[1][1] < np.inf]

        pd_h1.append(h1)
        pd_h0.append(h0)

        # Sum finite persistence only (infinite persistence = essential H1, counted separately)
        pers_sum_h1[wi] = sum(d - b for b, d in h1 if np.isfinite(d))

    # Adaptive sigma for K_PD — use median pairwise persistence distance
    pers_nonzero = pers_sum_h1[pers_sum_h1 > 0]
    sigma_w = float(np.median(np.abs(pers_nonzero - pers_nonzero.mean()))) if len(pers_nonzero) > 1 else 0.5
    sigma_w = max(sigma_w, 0.01)

    K_PD = compute_wasserstein_kernel(pd_h1, sigma=sigma_w)
    from utils import ensure_positive_definite
    K_PD = ensure_positive_definite(K_PD, name="K_PD")
    save_npz(str(out_dir / "persistence_kernel.npz"), K_PD=K_PD, pers_sum_h1=pers_sum_h1)

    logger.info("pers_sum_h1: min=%.4f, max=%.4f, nonzero=%d/%d",
                pers_sum_h1.min(), pers_sum_h1.max(),
                int(np.sum(pers_sum_h1 > 0)), n_windows)

    # Gap statistic KMeans — use pers_sum + gauge curvature (if available)
    logger.info("Running optimal K phase classification via Gap stat...")
    features = pers_sum_h1.reshape(-1, 1)

    try:
        ricci_data = np.load(str(Path(config.OUTPUTS_DIR) / "phase_E" / "ricci_tensor.npz"))
        ric_norm = np.linalg.norm(ricci_data["Ric"], axis=(1, 2))
        features = np.column_stack([features, ric_norm])
    except FileNotFoundError:
        pass

    # Standardise features for fair clustering
    from sklearn.preprocessing import StandardScaler
    features_scaled = StandardScaler().fit_transform(features)

    if OptimalK is not None:
        try:
            optimalK = OptimalK(parallel_backend='joblib')
            n_clusters = optimalK(features_scaled, cluster_array=np.arange(1, 7))
            gap_values = optimalK.gap_df["gap_value"].to_list()
        except Exception as exc:
            logger.warning("Gap stat failed: %s. Using elbow.", exc)
            n_clusters = 3
            gap_values = []
    else:
        logger.warning("gap-stat not installed. Defaulting to 3 phases.")
        n_clusters = 3
        gap_values = []

    kmeans = KMeans(n_clusters=int(n_clusters), random_state=config.SEED, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    with open(out_dir / "phase_classification.json", "w") as fh:
        json.dump({
            "optimal_k": int(n_clusters),
            "gap_values": gap_values,
            "labels": labels.tolist()
        }, fh, indent=2)
        
    logger.info("Topological Phase complete. Optimal phases = %d", n_clusters)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_topological_analysis()
