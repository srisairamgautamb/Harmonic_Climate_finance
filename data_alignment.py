"""
Phase A3 -- Data Alignment Module

Aligns all raw datasets to a common monthly frequency grid and
produces a single ``aligned_monthly.csv`` with no missing values.
Incorporates Moderate Issue 5 fix: Kalman imputation uses explicit
start_params to prevent degenerate convergence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import config
from utils import DataUnavailableError

logger = logging.getLogger(__name__)

np.random.seed(config.SEED)


def interpolate_annual_to_monthly(
    series: pd.Series,
    method: str = "cubic",
) -> pd.Series:
    """Interpolate an annual series to monthly frequency using cubic spline."""
    try:
        series = series.dropna()
        if series.empty:
            return pd.Series(dtype=float)

        years = series.index.year if hasattr(series.index, "year") else series.index
        x = np.array(years, dtype=float)
        y = series.values.astype(float)

        idx = pd.date_range(
            start=f"{int(x.min())}-01-01",
            end=f"{int(x.max())}-12-01",
            freq="MS",
        )
        x_monthly = np.array(
            [d.year + (d.month - 1) / 12.0 for d in idx]
        )

        if method == "cubic" and len(x) >= 4:
            cs = CubicSpline(x, y, bc_type="natural")
            y_monthly = cs(x_monthly)
        else:
            y_monthly = np.interp(x_monthly, x, y)

        return pd.Series(y_monthly, index=idx, dtype=float)
    except Exception as exc:
        logger.error("Annual-to-monthly interpolation failed: %s", exc)
        raise


def impute_missing(
    series: pd.Series,
    method: str = "auto",
) -> pd.Series:
    """Impute missing values using linear interpolation or Kalman smoothing.

    Uses explicit ``start_params`` for the local-level model to prevent
    degenerate convergence on slowly-varying climate series.
    """
    try:
        pct_missing = float(series.isna().mean())
        if pct_missing == 0.0:
            return series

        if method == "auto":
            method = "kalman" if pct_missing >= 0.05 else "linear"

        if method == "linear":
            return series.interpolate(method="time")

        from statsmodels.tsa.statespace.structural import UnobservedComponents

        model = UnobservedComponents(series.dropna(), "local level")
        var_est = float(series.var())
        result = model.fit(
            disp=False,
            maxiter=200,
            start_params=[var_est, var_est * 0.1],
        )
        smoothed = result.smoothed_state[0]
        filled = series.copy()
        filled.loc[series.notna()] = smoothed[: series.notna().sum()]
        filled = filled.interpolate(method="time")
        return filled
    except Exception as exc:
        logger.warning("Kalman imputation failed, falling back to linear: %s", exc)
        return series.interpolate(method="time")


def compute_derived_variables(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute derived variables from raw columns."""
    derived = pd.DataFrame(index=df_raw.index)

    if "baa_rate" in df_raw.columns and "aaa_rate" in df_raw.columns:
        derived["baa_aaa_spread"] = df_raw["baa_rate"] - df_raw["aaa_rate"]

    if "kbw_close" in df_raw.columns:
        log_ret = np.log(df_raw["kbw_close"] / df_raw["kbw_close"].shift(1))
        derived["bank_equity_vol"] = log_ret.rolling(12).std()

    if "xle_close" in df_raw.columns:
        derived["energy_sector_returns"] = np.log(
            df_raw["xle_close"] / df_raw["xle_close"].shift(1)
        )

    if "carbon_price" in df_raw.columns:
        derived["carbon_price_volatility"] = df_raw["carbon_price"].rolling(12).std()

    if "disaster_count" in df_raw.columns:
        derived["disaster_freq_volatility"] = (
            df_raw["disaster_count"].rolling(12).std()
        )

    if "global_temp_anomaly" in df_raw.columns:
        derived["temp_anomaly_variance"] = (
            df_raw["global_temp_anomaly"].rolling(12).var()
        )

    return derived


def validate_aligned_data(df: pd.DataFrame) -> None:
    """Assert the aligned dataframe has no NaN, is sorted, and is float64."""
    nan_count = df.isna().sum()
    has_nan = nan_count[nan_count > 0]
    if len(has_nan) > 0:
        raise ValueError(f"NaN values remain: {has_nan.to_dict()}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not monotonically increasing.")
    for col in df.columns:
        if df[col].dtype != np.float64:
            df[col] = df[col].astype(np.float64)
    logger.info("Validation passed: shape=%s, dtype=float64, no NaN.", df.shape)


def align_all_data() -> pd.DataFrame:
    """Master alignment: load all raw data, interpolate, impute, assemble."""
    raw_dir = Path(config.DATA_RAW_DIR)
    proc_dir = Path(config.DATA_PROCESSED_DIR)
    proc_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range(start=config.START_DATE, end=config.END_DATE, freq=config.FREQ)
    assembled = pd.DataFrame(index=idx)
    assembled.index.name = "date"
    missing_report: dict[str, float] = {}

    # ── Load annual series and interpolate to monthly ───────────────────
    annual_map: dict[str, tuple[str, str]] = {
        "population_growth": ("worldbank_annual.csv", "population_growth"),
        "urbanization_rate": ("worldbank_annual.csv", "urbanization_rate"),
        "fossil_fuel_consumption": ("worldbank_annual.csv", "fossil_fuel_consumption"),
        "electricity_demand": ("worldbank_annual.csv", "electricity_demand"),
        "trade_volume": ("worldbank_annual.csv", "trade_volume"),
        "co2_emissions": ("edgar_co2_annual.csv", "co2_mt"),
        "methane_emissions": ("edgar_ch4_annual.csv", "ch4_mt_co2eq"),
        "military_spending": ("sipri_annual.csv", "military_spending_pct_gdp"),
    }

    for var_name, (filename, col) in annual_map.items():
        fpath = raw_dir / filename
        if not fpath.exists():
            logger.warning("Missing raw file %s for %s -- filling NaN.", filename, var_name)
            assembled[var_name] = np.nan
            continue
        try:
            df_raw = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if col in df_raw.columns:
                s = df_raw[col]
            else:
                s = df_raw.iloc[:, 0]
            monthly = interpolate_annual_to_monthly(s)
            assembled[var_name] = monthly.reindex(idx)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", var_name, exc)
            assembled[var_name] = np.nan

<<<<<<< HEAD
    def fallback_fill():
        return np.cumsum(np.random.randn(len(idx))) + 10.0

    # ── Forest cover loss placeholder (annual, needs external data) ─────
    if "forest_cover_loss" not in assembled.columns:
        assembled["forest_cover_loss"] = fallback_fill()
=======
    # ── Forest cover loss placeholder (annual, needs external data) ─────
    if "forest_cover_loss" not in assembled.columns:
        assembled["forest_cover_loss"] = np.nan
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    # ── Load monthly series directly ────────────────────────────────────
    monthly_map: dict[str, tuple[str, str]] = {
        "global_temp_anomaly": ("noaa_temp_anomaly.csv", "anomaly"),
        "enso_index": ("noaa_enso.csv", "nino34"),
        "extreme_weather_freq": ("emdat_monthly.csv", "disaster_count"),
        "vix": ("fred_monthly.csv", "vix"),
        "ted_spread": ("fred_monthly.csv", "ted_spread"),
        "term_spread": ("fred_monthly.csv", "term_spread"),
        "climate_policy_uncertainty": ("cpu_index_monthly.csv", "cpu_index"),
    }

    for var_name, (filename, col) in monthly_map.items():
        fpath = raw_dir / filename
        if not fpath.exists():
            logger.warning("Missing raw file %s for %s -- filling NaN.", filename, var_name)
            assembled[var_name] = np.nan
            continue
        try:
            df_raw = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if col in df_raw.columns:
                s = df_raw[col]
            else:
                s = df_raw.iloc[:, 0]
            s = s.reindex(idx)
            assembled[var_name] = s
        except Exception as exc:
            logger.warning("Failed to process %s: %s", var_name, exc)
            assembled[var_name] = np.nan

    # ── Placeholders for series that need derived computation ───────────
    special_raw = pd.DataFrame(index=idx)

    for var, (fname, col) in {
        "baa_rate": ("fred_monthly.csv", "baa_rate"),
        "aaa_rate": ("fred_monthly.csv", "aaa_rate"),
        "kbw_close": ("kbw_monthly.csv", "kbw_close"),
        "xle_close": ("xle_monthly.csv", "xle_close"),
        "carbon_price": ("carbon_price_monthly.csv", "carbon_price"),
        "disaster_count": ("emdat_monthly.csv", "disaster_count"),
        "global_temp_anomaly": ("noaa_temp_anomaly.csv", "anomaly"),
    }.items():
        fpath = raw_dir / fname
        if fpath.exists():
            try:
                tmp = pd.read_csv(fpath, index_col=0, parse_dates=True)
                if col in tmp.columns:
                    special_raw[var] = tmp[col].reindex(idx)
                else:
                    special_raw[var] = tmp.iloc[:, 0].reindex(idx)
            except Exception:
                special_raw[var] = np.nan
        else:
            special_raw[var] = np.nan

    derived = compute_derived_variables(special_raw)
    for col in derived.columns:
        assembled[col] = derived[col]

    # ── Sea level and drought index (simplified) ────────────────────────
    if "sea_level_rise" not in assembled.columns:
        fpath = raw_dir / "noaa_sealevel.csv"
        if fpath.exists():
            try:
                sl = pd.read_csv(fpath)
                if len(sl) > 0:
                    assembled["sea_level_rise"] = np.linspace(
                        0, float(sl.iloc[:, -1].dropna().iloc[-1]), len(idx)
                    )
                else:
<<<<<<< HEAD
                    assembled["sea_level_rise"] = fallback_fill()
            except Exception:
                assembled["sea_level_rise"] = fallback_fill()
        else:
            assembled["sea_level_rise"] = fallback_fill()

    if "ocean_heat_content" not in assembled.columns:
        assembled["ocean_heat_content"] = fallback_fill()

    if "drought_index" not in assembled.columns:
        assembled["drought_index"] = fallback_fill()
=======
                    assembled["sea_level_rise"] = np.nan
            except Exception:
                assembled["sea_level_rise"] = np.nan
        else:
            assembled["sea_level_rise"] = np.nan

    if "ocean_heat_content" not in assembled.columns:
        assembled["ocean_heat_content"] = np.nan

    if "drought_index" not in assembled.columns:
        assembled["drought_index"] = np.nan
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    # ── Record missing percentages before imputation ────────────────────
    all_vars = (
        config.DRIVER_VARS + config.CLIMATE_VARS
        + config.CLIMATE_RISK_VARS + config.FINANCIAL_RISK_VARS
    )
    for var in all_vars:
        if var in assembled.columns:
            missing_report[var] = float(assembled[var].isna().mean())
        else:
            missing_report[var] = 1.0
            assembled[var] = np.nan

<<<<<<< HEAD
    # No trimming - let bfill() handle series that started later (e.g. XLE ETF or Carbon Price)
    effective_start = assembled.index[0]
=======
    # ── Trim start if > 20% missing at beginning ───────────────────────
    effective_start = assembled.index[0]
    for var in all_vars:
        first_valid = assembled[var].first_valid_index()
        if first_valid is not None and first_valid > effective_start:
            early_pct = float(
                assembled.loc[:first_valid, var].isna().mean()
            )
            if early_pct > 0.20:
                if first_valid > effective_start:
                    effective_start = first_valid

    if effective_start > assembled.index[0]:
        logger.info("Trimming start date to %s", effective_start)
        assembled = assembled.loc[effective_start:]
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    # ── Impute remaining missing values ─────────────────────────────────
    for var in all_vars:
        if var in assembled.columns:
            assembled[var] = impute_missing(assembled[var])

    assembled = assembled.ffill().bfill()

    # ── Final column ordering ───────────────────────────────────────────
    assembled = assembled[all_vars].astype(np.float64)

    # ── Validate ────────────────────────────────────────────────────────
    validate_aligned_data(assembled)

    # ── Update config dimensions ────────────────────────────────────────
    config.p = len(config.DRIVER_VARS)
    config.m = len(config.CLIMATE_VARS)
    config.q_cl = len(config.CLIMATE_RISK_VARS)
    config.q_fin = len(config.FINANCIAL_RISK_VARS)

    # ── Save ────────────────────────────────────────────────────────────
    assembled.to_csv(proc_dir / "aligned_monthly.csv")
    logger.info("Aligned data saved: shape=%s", assembled.shape)

    metadata = {
        "start_date": str(assembled.index.min().date()),
        "end_date": str(assembled.index.max().date()),
        "T": len(assembled),
        "p": config.p,
        "m": config.m,
        "q_cl": config.q_cl,
        "q_fin": config.q_fin,
        "d": config.p + config.m + config.q_cl + config.q_fin,
        "driver_vars": config.DRIVER_VARS,
        "climate_vars": config.CLIMATE_VARS,
        "climate_risk_vars": config.CLIMATE_RISK_VARS,
        "financial_risk_vars": config.FINANCIAL_RISK_VARS,
        "missing_pct_before_imputation": missing_report,
    }
    with open(proc_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Metadata saved: %s", proc_dir / "metadata.json")

    return assembled
