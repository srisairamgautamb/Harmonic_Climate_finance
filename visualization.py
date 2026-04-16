"""
Phase J1 -- Visualisation

Ten publication-quality figures covering data overview, spectral
heatmaps, Granger causality, phase lag, transmission fit, harmonic
potential, kernel matrices, model comparison, OOS forecast, and
dominant frequencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)

_STYLE = "seaborn-v0_8-whitegrid"
_FIGSIZE = (10, 6)
_DPI = 150


def _apply_style() -> None:
    try:
        plt.style.use(_STYLE)
    except OSError:
        plt.style.use("ggplot")


def plot_data_overview(Z_hat: pd.DataFrame) -> None:
    """Figure 1: 4x2 panel of key variables from each block."""
    _apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=_DPI)
    keys = [
        "population_growth", "co2_emissions",
        "global_temp_anomaly", "enso_index",
        "carbon_price_volatility", "climate_policy_uncertainty",
        "vix", "baa_aaa_spread",
    ]
    for i, key in enumerate(keys):
        ax = axes[i // 4, i % 4]
        if key in Z_hat.columns:
            ax.plot(Z_hat.index, Z_hat[key], linewidth=0.8)
        ax.set_title(key.replace("_", " ").title(), fontsize=9)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig1_data_overview.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_spectral_heatmap(
    S_all: np.ndarray,
    freqs: np.ndarray,
    var_names: list[str],
) -> None:
    """Figure 2: power spectrum heatmap for 4 key variables."""
    _apply_style()
    keys = ["co2_emissions", "global_temp_anomaly",
            "carbon_price_volatility", "vix"]
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE, dpi=_DPI)
    for idx, key in enumerate(keys):
        ax = axes[idx // 2, idx % 2]
        if key in var_names:
            vi = var_names.index(key)
            power = np.log10(np.abs(S_all[:, vi, vi, :].real) + 1e-12)
            ax.imshow(
                power, aspect="auto", origin="lower",
                extent=[float(freqs[0]), float(freqs[-1]), 0, power.shape[0]],
            )
            ax.set_xlabel("Frequency (cycles/month)", fontsize=8)
            ax.set_ylabel("Window centre", fontsize=8)
        ax.set_title(key.replace("_", " ").title(), fontsize=9)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig2_spectral_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_granger_causality(
    GC_all: np.ndarray,
    freqs: np.ndarray,
    pair_names: list[str],
) -> None:
    """Figure 3: spectral Granger causality by frequency band."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE, dpi=_DPI)
    display_pairs = pair_names[:4]
    periods = 1.0 / np.where(freqs > 0, freqs, 1e-12)
    for idx, name in enumerate(display_pairs):
        ax = axes[idx // 2, idx % 2]
        ax.plot(periods, GC_all[idx], linewidth=0.9)
        ax.axvspan(6, 24, alpha=0.1, color="blue", label="Short-run")
        ax.axvspan(24, 96, alpha=0.1, color="green", label="Medium-run")
        ax.set_xlabel("Period (months)", fontsize=8)
        ax.set_ylabel("GC(omega)", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(2, 200)
        ax.set_xscale("log")
        if idx == 0:
            ax.legend(fontsize=6)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig3_granger_causality.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_phase_lag(lag_all: np.ndarray, freqs: np.ndarray) -> None:
    """Figure 4: phase lead-lag heatmap."""
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    avg_lag = np.mean(lag_all, axis=0)
    if avg_lag.ndim >= 2:
        data = avg_lag[:, 0, :] if avg_lag.ndim == 3 else avg_lag
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", origin="lower")
        ax.set_xlabel("Frequency bin", fontsize=9)
        ax.set_ylabel("Variable pair", fontsize=9)
        plt.colorbar(im, ax=ax, label="Lag (months)")
    ax.set_title("Phase Lead-Lag by Frequency", fontsize=10)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig4_phase_lag.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_transmission_fit(
    T_empirical: np.ndarray,
    T_predicted: np.ndarray,
) -> None:
    """Figure 5: T_hat vs T_pred for each financial output."""
    _apply_style()
    q_fin = T_predicted.shape[1] if T_predicted.ndim >= 2 else 1
    fig, axes = plt.subplots(min(q_fin, 3), 1, figsize=(10, 8), dpi=_DPI)
    if q_fin == 1:
        axes = [axes]
    for k in range(min(q_fin, 3)):
        ax = axes[k]
        t_emp = np.real(np.mean(T_empirical[:, :, k, 0], axis=1)) if T_empirical.ndim >= 4 else T_empirical[:, k]
        t_pred = T_predicted[:, k, 0] if T_predicted.ndim >= 3 else T_predicted[:, k]
        ax.plot(t_emp, label="Empirical", linewidth=0.8, alpha=0.7)
        ax.plot(t_pred, label="Predicted", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Financial output {k}", fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig5_transmission_fit.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_harmonic_potential(Phi_values: np.ndarray) -> None:
    """Figure 6: harmonic potential over time for each output."""
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    if Phi_values.ndim >= 2:
        for k in range(min(Phi_values.shape[0], 6)):
            ax.plot(Phi_values[k].flatten()[:100], label=f"Phi_{k}", linewidth=0.8)
    ax.set_xlabel("Window centre", fontsize=9)
    ax.set_ylabel("Potential value", fontsize=9)
    ax.set_title("Harmonic Potential by Output Dimension", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig6_harmonic_potential.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_kernels(
    K_Harm: np.ndarray,
    K_QG: np.ndarray,
    K_HQG: np.ndarray,
) -> None:
    """Figure 7: kernel matrix heatmaps."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=_DPI)
    for ax, K, title in zip(axes, [K_Harm, K_QG, K_HQG],
                             ["K_Harm", "K_QG", "K_HQG"]):
        im = ax.imshow(K, aspect="auto", cmap="viridis")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig7_kernels.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
<<<<<<< HEAD
    """Figure 8: Publication-quality model comparison with fixed 7-model baseline."""
    _apply_style()

    # Define canonical order to ensure chart consistency
    canonical_order = ["LSR", "RBF-SVM", "RF", "LSTM", "VAR", "HQG-KRR", "HQG-SVM"]
    
    # Reindex the dataframe to ensure all models are listed in order
    comparison_df = comparison_df.set_index("model").reindex(canonical_order).reset_index()
    
    hqg_names = {"HQG-KRR", "HQG-SVM"}
    models = comparison_df["model"].tolist()
    is_hqg = [m in hqg_names for m in models]

    # Color scheme: classical = dark slate, HQG = crimson gold high-contrast
    colors_rmse = ["#C0392B" if h else "#2C3E50" for h in is_hqg]
    colors_f1 = ["#E74C3C" if h else "#34495E" for h in is_hqg]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    x = np.arange(len(models))

    # ── Subtitle for Hardware Transparency ──
    fig.suptitle("Quantum-Geometric Risk Transmission Benchmarks\n"
                 "Hardware Status: Qiskit Aer (Simulator)", 
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Panel A: RMSE ──
    if "rmse_vix" in comparison_df.columns:
        vals = comparison_df["rmse_vix"].fillna(0).values
        bars = axes[0].bar(x, vals, color=colors_rmse, edgecolor="white", alpha=0.9)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=35, ha="right", fontsize=9, fontweight="bold")
        axes[0].set_ylabel("RMSE (VIX, Standardised)", fontsize=11)
        axes[0].set_title("(a) VIX Prediction Error", fontsize=12, fontweight="bold")
        
        # Reference line at Best Classical
        classical_rmse = [v for v, h in zip(vals, is_hqg) if not h and v > 0]
        if classical_rmse:
            best_classical = min(classical_rmse)
            axes[0].axhline(y=best_classical, color="#7F8C8D", linestyle="--", alpha=0.6)

    # ── Panel B: F1-Score ──
    if "f1_stress" in comparison_df.columns:
        vals = comparison_df["f1_stress"].fillna(0).values
        bars = axes[1].bar(x, vals, color=colors_f1, edgecolor="white", alpha=0.9)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=35, ha="right", fontsize=9, fontweight="bold")
        axes[1].set_ylabel("F1-Score (Stress Detection)", fontsize=11)
        axes[1].set_title("(b) Market Stress Classification", fontsize=12, fontweight="bold")

    # Legend and Layout
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#C0392B", label="HQG (Quantum-Geometric)"),
        Patch(facecolor="#2C3E50", label="Classical Benchmarks"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2, 
               bbox_to_anchor=(0.5, 0.95), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig8_model_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 8: %s", out)
=======
    """Figure 8: grouped bar chart of model performance."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=_DPI)
    models = comparison_df["model"].tolist()
    x = np.arange(len(models))

    if "rmse_vix" in comparison_df.columns:
        vals = comparison_df["rmse_vix"].fillna(0).values
        axes[0].bar(x, vals, color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("RMSE (VIX)", fontsize=9)
        axes[0].set_title("VIX Prediction RMSE", fontsize=10)

    if "f1_stress" in comparison_df.columns:
        vals = comparison_df["f1_stress"].fillna(0).values
        axes[1].bar(x, vals, color="coral")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("F1 Score", fontsize=9)
        axes[1].set_title("Stress Classification F1", fontsize=10)

    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig8_model_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60


def plot_oos_forecast(
    y_true: np.ndarray,
    y_pred_dict: dict[str, np.ndarray],
    dates: np.ndarray,
) -> None:
    """Figure 9: out-of-sample VIX forecast comparison."""
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(dates, y_true, label="Actual", linewidth=1.2, color="black")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, (name, yp) in enumerate(y_pred_dict.items()):
        ax.plot(dates[:len(yp)], yp, label=name, linewidth=0.8,
                color=colors[i % len(colors)])

    crisis = [
        ("2008-09", "2009-06", "GFC"),
        ("2010-05", "2012-06", "EU Crisis"),
        ("2020-02", "2020-06", "COVID"),
    ]
    for start, end, label in crisis:
        try:
            ax.axvspan(
                pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.1, color="red",
            )
        except Exception:
            pass

    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("VIX (standardised)", fontsize=9)
    ax.set_title("Out-of-Sample VIX Forecast", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig9_oos_forecast.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_dominant_frequencies(
    GC_all: np.ndarray,
    freqs: np.ndarray,
    pair_names: list[str],
) -> None:
    """Figure 10: dominant frequencies with economic period annotations."""
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    for pi, name in enumerate(pair_names):
        peak_idx = int(np.argmax(GC_all[pi]))
        peak_freq = freqs[peak_idx]
        period = 1.0 / peak_freq if peak_freq > 0 else 0.0
        ax.barh(pi, period, color="steelblue", height=0.6)
        ax.text(period + 1, pi, f"{period:.0f} months", fontsize=7, va="center")
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names, fontsize=8)
    ax.set_xlabel("Dominant Period (months)", fontsize=9)
    ax.set_title("Dominant Frequencies per Causal Pair", fontsize=10)
    plt.tight_layout()
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "figures" / "fig10_dominant_frequencies.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)
def save_flowchart_mermaid() -> None:
    """Saves a mermaid flowchart description of the pipeline."""
    mermaid_code = """graph TD
    A[Raw Data] --> B[Spectral Analysis]
    B --> C[Geometric Map]
    C --> D[Quantum Engine]
    D --> E[Risk Prediction]
    """
    out = Path(config.OUTPUTS_DIR) / "phase_J" / "pipeline_flowchart.md"
    with open(out, "w") as f:
        f.write("# Project Pipeline Flowchart\\n\\n```mermaid\\n")
        f.write(mermaid_code)
        f.write("```\\n")
    logger.info("Saved %s", out)


def run_all_visualizations() -> None:
    """Phase J1 entry point."""
    fig_dir = Path(config.OUTPUTS_DIR) / "phase_J" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    out_base = Path(config.OUTPUTS_DIR)

    save_flowchart_mermaid()

    Z_hat = pd.read_csv(proc_dir / "Z_hat.csv", index_col=0, parse_dates=True)
    plot_data_overview(Z_hat)

    spec = np.load(str(out_base / "phase_B" / "spectral_features.npz"), allow_pickle=True)
    S_UU = spec["S_UU"]
    freqs = spec["freqs"]
    U_vars = list(pd.read_csv(proc_dir / "U_hat.csv", index_col=0, nrows=0).columns)
    plot_spectral_heatmap(S_UU, freqs, U_vars)
    plot_phase_lag(spec["lag_UF"], freqs)

    caus = np.load(str(out_base / "phase_C" / "causality_results.npz"), allow_pickle=True)
    pair_file = out_base / "phase_C" / "pair_names.json"
    pair_names = json.load(open(pair_file)) if pair_file.exists() else []
    plot_granger_causality(caus["GC_all"], caus["freqs"], pair_names)
    plot_dominant_frequencies(caus["GC_all"], caus["freqs"], pair_names)

    T_emp_path = out_base / "phase_D" / "T_hat_empirical.npz"
    T_pred_path = out_base / "phase_E" / "T_pred.npz"
    if T_emp_path.exists() and T_pred_path.exists():
        T_emp = np.load(str(T_emp_path), allow_pickle=True)["T_hat_empirical"]
        T_pred = np.load(str(T_pred_path), allow_pickle=True)["T_pred"]
        plot_transmission_fit(T_emp, T_pred)

    phi_path = out_base / "phase_E" / "Phi_hat.npz"
    if phi_path.exists():
        phi_data = np.load(str(phi_path), allow_pickle=True)["Phi_hat"]
        plot_harmonic_potential(phi_data)

    for kf in ["K_Harm.npz", "K_QG.npz", "K_HQG.npz"]:
        kp = out_base / "phase_F" / kf
        if not kp.exists():
            break
    else:
        Kh = np.load(str(out_base / "phase_F" / "K_Harm.npz"), allow_pickle=True)["K"]
        Kq = np.load(str(out_base / "phase_F" / "K_QG.npz"), allow_pickle=True)["K"]
        Khqg = np.load(str(out_base / "phase_F" / "K_HQG.npz"), allow_pickle=True)["K"]
        plot_kernels(Kh, Kq, Khqg)

    comp_path = out_base / "phase_H" / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        plot_model_comparison(comp)

    # Figure 9: OOS Forecast
    y_pred_dict = {}  # {name: (dates, y_pred)}
<<<<<<< HEAD
    y_test_full = None
    y_test_dates_full = None
    
    # Get full test dates
    test_size = 79 # From log: Preprocessing complete. T'=394, train=315, test=79
    y_test_dates_full = Z_hat.index[-test_size:]
=======
    y_actual = None
    y_actual_dates = None
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    # Try loading various model predictions
    for phase, name, fname in [
        ("phase_H", "LSTM", "lstm_predictions.npz"),
        ("phase_H", "VAR", "var_predictions.npz"),
        ("phase_I", "HQG-KRR", "hqg_krr_predictions.npz"),
    ]:
        ppath = out_base / phase / fname
        if ppath.exists():
            data = np.load(str(ppath), allow_pickle=True)
            yp = data["y_pred"]
            yt = data["y_true"]
            
<<<<<<< HEAD
            if y_test_full is None:
                y_test_full = yt # Assumed to be the full test set
            
            # Extract dates for this specific prediction
            if len(yp) == test_size:
                dts = y_test_dates_full
=======
            # Extract dates for test set
            test_idx = data.get("test_idx", None)
            if test_idx is not None:
                dts = Z_hat.index[test_idx]
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
            else:
                dts = Z_hat.index[-len(yp):]
            
            y_pred_dict[name] = (dts, yp)
<<<<<<< HEAD

    if y_test_full is not None:
        _apply_style()
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        
        # Ensure actuals match the dates
        if len(y_test_full) != len(y_test_dates_full):
             # Fallback if y_test_full in npz is not the full test set
             y_test_full_dates_adj = Z_hat.index[-len(y_test_full):]
        else:
             y_test_full_dates_adj = y_test_dates_full
             
        ax.plot(y_test_full_dates_adj, y_test_full, label="Actual", linewidth=1.2, color="black")
        
        # Color scheme matching Fig 8 (HQG is crimson, Classical are cool tones)
        color_map = {
            "HQG-KRR": "#C0392B",  # Crimson
            "HQG-SVM": "#E74C3C",  # Lighter Red
            "LSTM": "#2980B9",     # Blue
            "VAR": "#7F8C8D",      # Gray
            "RBF-SVM": "#27AE60",  # Green
            "LSR": "#F39C12",      # Orange
            "RF": "#8E44AD"        # Purple
        }
        
        for name, (dts, yp) in y_pred_dict.items():
            # One more safety check for model-specific dts/yp
            if len(dts) != len(yp):
                dts = dts[:len(yp)] # Truncate if mismatch
            
            color = color_map.get(name, "black")
            linewidth = 1.5 if "HQG" in name else 0.8
            alpha = 1.0 if "HQG" in name else 0.7
            
            ax.plot(dts, yp, label=name, linewidth=linewidth, color=color, alpha=alpha)
=======
            if y_actual is None:
                y_actual = yt
                y_actual_dates = dts

    if y_actual is not None:
        # Use the actual dates from the first model as base, but plot others on their own
        _apply_style()
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        ax.plot(y_actual_dates, y_actual, label="Actual", linewidth=1.2, color="black")
        
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, (name, (dts, yp)) in enumerate(y_pred_dict.items()):
            ax.plot(dts, yp, label=name, linewidth=0.8, color=colors[i % len(colors)])
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

        crisis = [
            ("2008-09", "2009-06", "GFC"),
            ("2010-05", "2012-06", "EU Crisis"),
            ("2020-02", "2020-06", "COVID"),
        ]
        for start, end, label in crisis:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color="red")
            except Exception: pass

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("VIX (standardised)", fontsize=9)
        ax.set_title("Out-of-Sample VIX Forecast", fontsize=10)
        ax.legend(fontsize=7)
        plt.tight_layout()
        out = fig_dir / "fig9_oos_forecast.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved %s", out)

    logger.info("Phase J complete.")
