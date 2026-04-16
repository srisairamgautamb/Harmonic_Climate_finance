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


def run_topological_analysis() -> None:
    out_dir = Path(config.OUTPUTS_DIR) / "phase_Topo"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    spectral_data = np.load(str(Path(config.OUTPUTS_DIR) / "phase_B" / "spectral_features.npz"))
    Coh_UU = spectral_data["Coh_UU"]  # [n_windows, d_U, d_U, n_freq]
    
    n_windows = Coh_UU.shape[0]
    pd_h1 = []
    pd_h0 = []
    
    logger.info("Computing Vietoris-Rips persistence over %d windows...", n_windows)
    
    pers_sum_h1 = np.zeros(n_windows)
    
    for wi in range(n_windows):
        dist_mat = 1.0 - np.mean(Coh_UU[wi], axis=-1)
        np.fill_diagonal(dist_mat, 0.0)
        dist_mat = np.clip(dist_mat, 0.0, 1.0)
        
        rips = gudhi.RipsComplex(distance_matrix=dist_mat, max_edge_length=1.0)
        st = rips.create_simplex_tree(max_dimension=2)
        diag = st.persistence()
        
        h1 = [p[1] for p in diag if p[0] == 1]
        h0 = [p[1] for p in diag if p[0] == 0 and p[1][1] < np.inf]
        
        pd_h1.append(h1)
        pd_h0.append(h0)
        
        pers_sum_h1[wi] = sum([d - b for b, d in h1])

    sigma_w = 0.5
    K_PD = compute_wasserstein_kernel(pd_h1, sigma=sigma_w)
    
    save_npz(str(out_dir / "persistence_kernel.npz"), K_PD=K_PD, pers_sum_h1=pers_sum_h1)
    
    # Gap statistic KMeans
    logger.info("Running optimal K phase classification via Gap stat...")
    
    features = pers_sum_h1.reshape(-1, 1)
    
    try:
        ricci_data = np.load(str(Path(config.OUTPUTS_DIR) / "phase_E" / "ricci_tensor.npz"))
        ric_norm = np.linalg.norm(ricci_data["Ric"], axis=(1, 2))
        features = np.column_stack([features, ric_norm])
    except FileNotFoundError:
        pass
        
    if OptimalK is not None:
        optimalK = OptimalK(parallel_backend='joblib')
        n_clusters = optimalK(features, cluster_array=np.arange(1, 6))
        gap_values = optimalK.gap_df["gap_value"].to_list()
    else:
        logger.warning("gap-stat not installed. Defaulting to 2 phases.")
        n_clusters = 2
        gap_values = []
    
    # Extract KMeans clusters
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=config.SEED)
    labels = kmeans.fit_predict(features)
    
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
