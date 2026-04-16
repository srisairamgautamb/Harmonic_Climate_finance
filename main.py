"""
Master pipeline runner for the Spectral Climate-Financial Risk
Transmission framework.

Usage:
    python main.py --phases ALL
    python main.py --phases B1 C1
    python main.py --skip A2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable


def setup_logging() -> None:
    """Configure root logger to file and stdout."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/main.log"),
            logging.StreamHandler(),
        ],
    )


def create_directory_structure() -> None:
    """Create all required directories."""
    dirs = [
        "data/raw", "data/processed",
        "outputs/phase_B", "outputs/phase_C",
        "outputs/phase_D", "outputs/phase_E",
        "outputs/phase_F", "outputs/phase_G",
        "outputs/phase_H", "outputs/phase_I",
        "outputs/phase_J/figures", "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logging.info("Directory structure created.")


def run_phase(name: str, fn: Callable[[], None]) -> dict:
    """Time and execute a single phase, returning status metadata."""
    logging.info("===== STARTING PHASE %s =====", name)
    t0 = time.time()
    try:
        fn()
        elapsed = round(time.time() - t0, 2)
        logging.info("===== PHASE %s COMPLETE in %ss =====", name, elapsed)
        return {"phase": name, "status": "ok",
                "elapsed_seconds": elapsed, "error_msg": None}
    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        logging.error(
            "===== PHASE %s FAILED after %ss: %s =====",
            name, elapsed, exc, exc_info=True,
        )
        return {"phase": name, "status": "error",
                "elapsed_seconds": elapsed, "error_msg": str(exc)}


def run_pipeline(phases: list[str]) -> None:
    """Run the full pipeline or a subset of phases."""
    import classical_baselines
    import data_alignment
    import data_download
    import harmonic_potential
    import hqg_models
    import kernel_construction
    import preprocessing
    import quantum_embedding
    import spectral_causality
    import spectral_estimation
    import var_spectral_param
    import visualization
    import topological_analysis

    ALL_PHASES: dict[str, Callable[[], None]] = {
        "A2": data_download.download_all_data,
        "A3": data_alignment.align_all_data,
        "A4": preprocessing.preprocess_all,
        "B1": spectral_estimation.run_spectral_estimation,
        "C1": spectral_causality.run_causality_analysis,
        "D1": var_spectral_param.run_var_spectral_param,
        "E1": harmonic_potential.run_harmonic_potential,
        "F1": kernel_construction.run_kernel_construction,
        "G1": quantum_embedding.run_quantum_embedding,
        "H1": classical_baselines.run_classical_baselines,
        "T1": topological_analysis.run_topological_analysis,
        "I1": hqg_models.run_hqg_models,
        "J1": visualization.run_all_visualizations,
    }

    phases_to_run = list(ALL_PHASES.keys()) if "ALL" in phases else phases

    results: list[dict] = []
    for code in phases_to_run:
        if code not in ALL_PHASES:
            logging.warning("Unknown phase code: %s. Skipping.", code)
            continue
        result = run_phase(code, ALL_PHASES[code])
        results.append(result)
        if result["status"] == "error" and code in ("A2", "A3", "A4"):
            logging.error("Critical data phase failed. Stopping pipeline.")
            break

    summary_path = "outputs/pipeline_run_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logging.info("Pipeline run summary saved to %s", summary_path)

    print("\n" + "=" * 60)
    print(f"{'PHASE':<10} {'STATUS':<10} {'TIME (s)':<12} {'ERROR'}")
    print("=" * 60)
    for r in results:
        err = (r["error_msg"] or "None")[:40]
        print(f"{r['phase']:<10} {r['status']:<10} {r['elapsed_seconds']:<12} {err}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spectral Climate-Financial Risk Transmission Pipeline",
    )
    parser.add_argument(
        "--phases", nargs="+", default=["ALL"],
        help="Phase codes to run (default: ALL)",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Phase codes to skip",
    )
    args = parser.parse_args()

    setup_logging()
    create_directory_structure()

    ALL_CODES = ["A2", "A3", "A4", "B1", "C1", "D1",
                 "E1", "F1", "G1", "H1", "T1", "I1", "J1"]
    phases = args.phases
    if args.skip:
        base = ALL_CODES if "ALL" in phases else phases
        phases = [p for p in base if p not in args.skip]

    run_pipeline(phases)
