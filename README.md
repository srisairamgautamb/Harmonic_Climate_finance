# Spectral Climate-Financial Risk Transmission: A Harmonic Quantum Machine Learning Framework

Sri Sairam Gautam B., Divyendu Kashyap
School of Engineering, Jawaharlal Nehru University

This repository contains the implementation behind the paper of the same title. It models the transmission of climate and socioeconomic shocks into financial risk as a harmonic process on a statistical manifold, and uses the resulting geometric and spectral structure to build a kernel for forecasting financial risk indicators.

## Overview

Conventional time-domain econometric models treat climate-financial linkages as locally linear and stationary. The framework here combines four components:

1. Spectral analysis of rolling-window multivariate dynamics, including frequency-domain Granger causality and phase lag.
2. Information geometry: local VAR dynamics are parameterized by Whittle likelihood and embedded on a manifold with the Fisher-Rao metric.
3. Harmonic analysis: a graph Laplacian over windows, weighted by Fisher-Rao distance, yields harmonic basis functions and a potential field obtained from a discrete Dirichlet boundary value problem.
4. The Harmonic Quantum Geometry (HQG) kernel, which combines a harmonic kernel, a geodesic kernel on Fisher-Rao distances, and a quantum state-fidelity kernel from a 6-qubit angle-encoded circuit, and is used in kernel ridge regression (HQG-KRR) and SVM classification (HQG-SVM).

The quantum component is simulated classically with a statevector simulator. It is used as an expressive feature map, not as a claim of computational quantum advantage.

## Formulation

The system is a multivariate process $Z_t = (D_t, C_t, R^{cl}_t, R^{fin}_t)$ of socioeconomic drivers, climate indicators, climate risk proxies and financial risk indicators, assumed to follow the causal ordering $D \to C \to R^{cl} \to R^{fin}$. Within each 60-month window the dynamics are approximated by a VAR process, and the spectral density matrix $S_Z(\omega)$ is estimated from it.

Spectral causality from $X$ to $Y$ at frequency $\omega$:

$$\mathcal{F}_{X \to Y}(\omega) = \log \frac{\det S_{YY}(\omega)}{\det S_{YY|X}(\omega)}$$

Window graph weights from Fisher-Rao distance, and the Laplacian whose harmonic condition defines the risk potential $\Phi$:

$$W_{ij} = \exp\left(-\frac{d^2(\theta_i, \theta_j)}{\sigma^2}\right), \qquad L = D - W, \qquad L\Phi = 0$$

HQG kernel and forecaster:

$$K_{HQG} = \lambda_h K_{harm} + \lambda_g K_{geo} + \lambda_q K_{quant}, \qquad \hat{y} = K_{HQG}(K_{HQG} + \lambda I)^{-1} y$$

The weights are non-negative and sum to one. They are selected by grid search on kernel-target alignment over training windows, and the adaptive variant re-optimizes them every 5 walk-forward steps on an expanding window. $K_{geo}$ is a Matern kernel on Fisher-Rao geodesic distances, and $K_{quant}(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$.

## Data

Monthly observations from 1990 to 2023 for 25 variables in four blocks:

| Block | Variables | Examples |
|:---|:---:|:---|
| Socioeconomic drivers | 9 | population growth, fossil fuel consumption, CO2 emissions, trade volume |
| Physical climate indicators | 6 | temperature anomaly, ENSO index, sea level, drought index |
| Climate risk proxies | 4 | carbon price volatility, climate policy uncertainty |
| Financial risk indicators | 6 | VIX, Baa-Aaa credit spread, bank equity volatility |

Sources are FRED, World Bank APIs and NASA/NOAA climate datasets, retrieved programmatically. Preprocessing resamples to monthly frequency, applies linear interpolation and forward fill, log-differences and standardizes for approximate stationarity, and scales to zero mean and unit variance. Rolling 60-month windows with a one-month step give 348 windows.

## Pipeline

`main.py` runs the stages in order. Phase codes are accepted by `--phases` and `--skip`.

| Stage | Code | Module | Function |
|:---|:---:|:---|:---|
| 1. Data acquisition and preprocessing | A2, A3, A4 | `data_download.py`, `data_alignment.py`, `preprocessing.py` | Download, chronological alignment, imputation, stationarity transforms |
| 2. Spectral estimation | B1 | `spectral_estimation.py` | Rolling-window cross-spectral density matrices |
| 3. Spectral causality | C1 | `spectral_causality.py` | VAR fitting, spectral Granger causality, phase lag |
| 4. Fisher-Rao geometry | D1 | `var_spectral_param.py` | Whittle parameterization, Fisher-Rao metric |
| 5. Harmonic potential | E1 | `harmonic_potential.py` | Graph Laplacian eigendecomposition, Dirichlet problem, curvature proxy |
| 6. Kernel construction | F1 | `kernel_construction.py` | Harmonic and geodesic kernels, HQG combination |
| 7. Quantum embedding | G1 | `quantum_embedding.py` | 6-qubit angle encoding, fidelity kernel |
| 8. Classical baselines | H1 | `classical_baselines.py` | LSR, RBF-SVM, Random Forest, LSTM, VAR |
| Topological analysis | T1 | `topological_analysis.py` | Vietoris-Rips persistence, persistence-diagram kernel, gap-statistic regime clustering |
| 9. HQG models | I1 | `hqg_models.py` | HQG-KRR, adaptive HQG-KRR, HQG-SVM, statistical tests |
| 10. Visualization | J1 | `visualization.py` | Figure generation (14 PDFs) |

## Results

Forecasting target is next-period standardized VIX. The split is chronological with 296 training and 52 test observations. Values are from `outputs/phase_H/model_comparison.csv` and `outputs/phase_I/`.

| Model | RMSE | R^2 |
|:---|---:|---:|
| LSR | 1.018 | 0.191 |
| RBF-SVM | 1.039 | 0.158 |
| LSTM | 1.033 | 0.167 |
| Random Forest | 1.168 | -0.065 |
| VAR | 1.179 | -0.080 |
| HQG-KRR | 0.988 | 0.237 |
| HQG-KRR, adaptive weights | 0.963 | 0.276 |

- HQG-KRR has the lowest RMSE and highest R^2 of the evaluated models; the adaptive variant improves both further.
- Against LSR, the strongest baseline, the Diebold-Mariano test gives statistic -0.21 (p = 0.834). The Clark-West test for nested models gives 1.69 (one-sided p = 0.046). With 52 test observations the Diebold-Mariano test has limited power.
- Split conformal prediction intervals at 90% nominal level reach 91.9% empirical coverage on a calibration set of 74 windows, with half-width 1.45 in standardized units.
- Stress-regime classification with HQG-SVM gives F1 0.200, precision 0.167, recall 0.250 and AUC 0.531 on the test set. The LSR and RBF-SVM classifiers give F1 0.0 under the same protocol.

Spectral Granger causality between blocks, from `outputs/phase_C/causality_summary.csv`:

| Direction | Total causality | Peak period (months) |
|:---|---:|:---|
| $D \to R^{fin}$ | 3.91 | 7.6 |
| $D \to C$ | 3.11 | 7.6 |
| $C \to R^{fin}$ | 1.81 | 7.2 |
| $D \to R^{cl}$ | 1.78 | zero frequency |
| $R^{cl} \to R^{fin}$ | 1.37 | zero frequency |
| $C \to R^{cl}$ | 1.13 | 7.2 |

Socioeconomic drivers carry the largest total causal influence on financial risk indicators among the six block pairs, consistent with the layered ordering assumed in the framework.

## Scope and limitations

- Windows are treated as locally stationary, which may not capture structural breaks or long-term nonstationarity.
- The test set is short (52 months), and performance differences relative to LSR are modest. Results have been reported for a single chronological split.
- The full Fisher-Rao metric tensor is memory-intensive. The pipeline works with a Nystrom-reduced metric, and the curvature quantity is a scalar proxy computed from the log-determinant of a leading 64-dimensional block of the full metric, normalized by its 95th percentile. Its magnitude should be read as relative variation across windows, not absolute curvature.
- The quantum fidelity kernel is computed on the training block and on test-to-train pairs by statevector simulation. Test-to-test entries use an RBF approximation on the VAR parameters, which does not affect KRR predictions. Behavior on quantum hardware has not been evaluated.
- The HQG-SVM stress detector is weakly discriminative at this sample size and is reported as a preliminary result.

## Installation

Requires Python 3.10 or later and a FRED API key.

```bash
git clone https://github.com/srisairamgautamb/Harmonic_Climate_finance.git
cd Harmonic_Climate_finance
chmod +x environment_setup.sh
./environment_setup.sh
export FRED_API_KEY=<your key>
```

Standard runs need roughly 16 GB of memory.

## Usage

```bash
python main.py --phases ALL          # full pipeline
python main.py --phases B1 C1        # selected phases
python main.py --skip A2             # skip data download
python -m pytest tests               # unit tests
```

Run summaries are written to `outputs/pipeline_run_summary.json`, model results to JSON under `outputs/phase_*/`, and figures to `outputs/phase_J/figures/`.

## Repository layout

```text
config.py                 pipeline configuration and hyperparameters
main.py                   orchestrator
utils.py                  shared metrics, tests and I/O helpers
data_*.py, preprocessing.py
spectral_*.py, var_spectral_param.py
harmonic_potential.py, kernel_construction.py, quantum_embedding.py
topological_analysis.py, classical_baselines.py, hqg_models.py
visualization.py
tests/                    unit tests
data/processed/           aligned and transformed series
outputs/                  per-phase results and figures
```

## License

MIT License.
