"""
Configuration constants for the Spectral Climate-Financial Risk
Transmission pipeline.  No logic resides here.
"""

from __future__ import annotations

from pathlib import Path

# ── Time range ──────────────────────────────────────────────────────────
START_DATE: str = "1990-01-01"
END_DATE: str = "2023-12-31"
FREQ: str = "MS"

# ── Block dimensions (populated after data assembly) ────────────────────
p: int | None = None
m: int | None = None
q_cl: int | None = None
q_fin: int | None = None

# ── Variable names per block ────────────────────────────────────────────
DRIVER_VARS: list[str] = [
    "population_growth",
    "urbanization_rate",
    "fossil_fuel_consumption",
    "electricity_demand",
    "co2_emissions",
    "methane_emissions",
    "forest_cover_loss",
    "trade_volume",
    "military_spending",
]

CLIMATE_VARS: list[str] = [
    "global_temp_anomaly",
    "ocean_heat_content",
    "enso_index",
    "extreme_weather_freq",
    "drought_index",
    "sea_level_rise",
]

CLIMATE_RISK_VARS: list[str] = [
    "carbon_price_volatility",
    "climate_policy_uncertainty",
    "disaster_freq_volatility",
    "temp_anomaly_variance",
]

FINANCIAL_RISK_VARS: list[str] = [
    "vix",
    "baa_aaa_spread",
    "ted_spread",
    "bank_equity_vol",
    "energy_sector_returns",
    "term_spread",
]

# ── Directories ─────────────────────────────────────────────────────────
DATA_RAW_DIR: str = "data/raw/"
DATA_PROCESSED_DIR: str = "data/processed/"
OUTPUTS_DIR: str = "outputs/"
LOGS_DIR: str = "logs/"

# ── Data Mode ─────────────────────────────────────────────────────────
USE_SYNTHETIC_DATA: bool = False

# ── Spectral parameters ────────────────────────────────────────────────
WINDOW_SIZE: int = 60
WINDOW_STEP: int = 1
N_FREQ: int = 30
TAPER: str = "hann"

# ── VAR parameters ─────────────────────────────────────────────────────
VAR_MAX_LAG: int = 3

# ── Kernel parameters ──────────────────────────────────────────────────
C_MASS: float = 1.0
ALPHA_TEMPORAL: float = 0.1
ALPHA_QG: float = 1.0

# ── Quantum parameters ─────────────────────────────────────────────────
N_QUBITS: int = 6
N_LAYERS: int = 3

# ── Random seed ─────────────────────────────────────────────────────────
SEED: int = 42
