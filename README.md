# Harmonic Climate Finance: Spectral Climate-Financial Risk Transmission

<div align="center">

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Operational](https://img.shields.io/badge/Status-Operational-brightgreen.svg)]()
[![Pipeline: 10 Phases](https://img.shields.io/badge/Pipeline-10%20Phases-orange.svg)]()

</div>

---

## 📖 Executive Summary

The **Harmonic Climate Finance** project investigates a fundamental macroeconomic and financial question: **How do climate shocks (rising temperatures, extreme weather, droughts) transmit through the global economy and eventually manifest as financial market risk?**

This project models climate-financial risk transmission as a wave propagating across a curved economic manifold. By unifying **Spectral Analysis**, **Riemannian Geometry**, **Quantum-Inspired Computing**, and **Harmonic Analysis**, we present an operational 10-phase computational pipeline that ingests real-world socioeconomic, climate, and financial data to model, measure, and forecast systemic risks.

---

## 🧠 Core Mathematical Architecture

The pipeline is built on a novel, unified mathematical framework:

1. **Spectral Analysis (Frequency Domain):** We decompose 34 years of time-series data into continuous frequency bands to analyze lead-lag relationships and identifying high-frequency rapid transmission of climate shocks.
2. **Fisher-Rao Information Geometry:** We map the economy onto a curved statistical manifold. By computing the Fisher-Rao metric from local Vector Autoregressive (VAR) parameters, we define rigorous "distances" between different economic risk regimes.
3. **Harmonic Wave Equations:** We model risk transmission like sound travelling over a drum, building a Graph Laplacian on the economic manifold and solving Dirichlet boundary value problems to extract spatial and frequency "harmonics."
4. **Harmonic Quantum Geometry (HQG) Kernel:** We construct a hybrid kernel (`K_HQG`) that combines geometric distances (via Fisher-Rao), spatial harmonics, and quantum correlations (via state fidelity from a 6-qubit angle-encoded quantum circuit), using it to power Quantum Kernel Ridge Regression (QKR) for forecasting.

---

## 📊 Data Architecture

The pipeline operates on 34 years of monthly data (1990–2023) across 25 distinct variables, organized into a multi-block causal chain:

- 🌍 **Socioeconomic Drivers (9 variables):** e.g., Population Growth, Fossil Fuel Consumption, CO₂ Emissions, Trade Volume.
- 🌡️ **Physical Climate Indicators (6 variables):** e.g., Global Temperature Anomaly, ENSO Index, Sea Level Rise, Drought Index.
- 📉 **Climate Risk Proxies (4 variables):** e.g., Carbon Price Volatility, Climate Policy Uncertainty.
- 🏦 **Financial Market Risks (6 variables):** e.g., VIX (Volatility Index), Baa-Aaa Credit Spread, Bank Equity Volatility.

*Data is sourced programmatically via FRED (Federal Reserve), World Bank APIs, and NOAA/NASA datasets.*

---

## 🚀 The 10-Phase Pipeline

The system is fully automated through `main.py`, executing 10 distinct sequential phases:

| Phase | Module | Description |
|:---:|:---|:---|
| **A** | `data_download.py` & `preprocessing.py` | API downloads, chronological alignment, imputation, and stationarity transformations. |
| **B** | `spectral_estimation.py` | Rolling window (60-mo) cross-spectral density matrix computation. |
| **C** | `spectral_causality.py` | VAR(3) fitting and spectral Granger causality tests across all variable blocks. |
| **D** | `var_spectral_param.py` | Whittle MLE parameterization and empirical Fisher-Rao Riemannian metric tensor computation. |
| **E** | `harmonic_potential.py` | Graph Laplacian eigen-decomposition, Dirichlet BVP wave solving, and Ricci curvature analysis. |
| **F** | `kernel_construction.py` | Synthesis of Harmonic (`K_Harm`) and Quantum-Geometric (`K_QG`) matrices into the `K_HQG` kernel. |
| **G** | `quantum_embedding.py` | 6-qubit quantum circuit angle-encoding and simulated fidelity kernel generation. |
| **H** | `classical_baselines.py` | Traditional benchmarking against LSR, RBF-SVM, Random Forests, LSTM, and empirical VAR models. |
| **I** | `hqg_models.py` | Training and optimal hyperparameter grid-search of the HQG-SVM classifiers and HQG-KRR forecasters. |
| **J** | `visualization.py` | Automated generation of 14+ publication-quality PDF figures. |

---

## 💻 Installation & Usage

### 1. Prerequisites
Ensure you have **Python 3.14** installed. Memory requirements vary: standard runs require ~16GB RAM. Computing the complete unbypassed Fisher-Rao metric tensor object in Phase D may require 200GB+ RAM. 

### 2. Setup
```bash
git clone https://github.com/srisairamgautamb/Harmonic_Climate_finance.git
cd Harmonic_Climate_finance
chmod +x environment_setup.sh
./environment_setup.sh
```

### 3. Execution
To run the full pipeline, simply execute the main orchestrator:
```bash
python main.py
```
To run specific phases (e.g., Phase J for visualization):
```bash
python main.py --phases J1
```

---

## 📈 Key Findings & Output Metrics

- **Causality Topology:** Socioeconomic drivers exhibit the strongest, most immediate causal effect on financial risk (Spectral GC ≈ 7.2), with transmission tightly localized to high-frequency domains (periodicity ≈ 0.3-0.4 months). 
- **Topological Flatness:** The Ricci curvature of the economic manifold evaluates exactly to zero (`max|Ric| = 0.00`), indicating that the systemic risk wave propagates without damping.
- **Model Forensics (Diebold-Mariano):** The proposed formulation HQG-KRR model was empirically compared against standard macroeconomic autoregression (VAR). The DM test (`p-value = 0.650`) showed statistical equivalency in forecasting capability, while the HQG framework inherently offers deeper rigorous topological interpretability.

*Note: All output model parameters, numerical forecasts (JSON/NPZ), and publication-ready charts (PDF) are saved to the `outputs/` directory.*

---

## 📂 Project Structure

```text
Harmonic_Climate_finance/
├── config.py                 # Pipeline configuration and hyperparameters
├── main.py                   # Master orchestrator
├── *_models.py / *.py        # Core Phase A–J logic scripts
├── outputs/                  # Local directory for generated model state artifacts
│   ├── phase_H/              # Baseline benchmarks & evaluations
│   ├── phase_I/              # HQG metrics & optimal hyperparameters
│   └── phase_J/figures/      # Rendered publication-ready empirical figures
├── data/                     # Raw and processed CSV data (git-ignored)
└── requirements.txt          # Python dependencies
```

---

## 📝 License
This project is open-source and available under the terms of the MIT License.
