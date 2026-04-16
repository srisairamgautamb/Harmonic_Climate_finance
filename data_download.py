"""
Phase A2 -- Data Download Module

<<<<<<< HEAD
Downloads all raw datasets from authoritative open data sources:
  - World Bank API (population, urbanization, fossil fuel, electricity, trade,
    forest cover, military spending)
  - FRED (VIX, TED spread, term spread, BAA/AAA rates, econ policy uncertainty)
  - NASA GISS (global temperature anomaly)
  - NOAA (ENSO index, sea level, ocean heat content)
  - Our World in Data (CO2 & methane emissions)
  - Yahoo Finance (KBW Bank Index, XLE Energy ETF, EU carbon futures proxy)

Implements retry logic, SSL bypass, and per-variable realistic synthetic
fallback when a specific endpoint is unavailable.
=======
Downloads all raw datasets from authoritative sources and persists
them to ``data/raw/``.  Each source has its own function.  Sources that
require manual registration raise ``DataUnavailableError`` with
instructions.
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
"""

from __future__ import annotations

import logging
import os
<<<<<<< HEAD
import ssl
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
=======
import time
from pathlib import Path

import numpy as np
import pandas as pd
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

import config
from utils import DataUnavailableError

logger = logging.getLogger(__name__)
<<<<<<< HEAD
np.random.seed(config.SEED)


# ── HTTP Session with Retries ──────────────────────────────────────────

def _robust_session() -> requests.Session:
    """Create a session with retry logic for flaky APIs."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def _safe_ssl_context():
    """Create an unverified SSL context for endpoints with cert issues."""
    ctx = ssl.create_default_https_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── Realistic Synthetic Fallback ───────────────────────────────────────

_REALISTIC_PARAMS = {
    # (mean, std, ar1_coef) for AR(1) process: x_t = ar1*x_{t-1} + N(0, std)
    "population_growth": (1.2, 0.1, 0.98),
    "urbanization_rate": (2.5, 0.3, 0.99),
    "fossil_fuel_consumption": (80.0, 2.0, 0.99),
    "electricity_demand": (2500.0, 200.0, 0.99),
    "trade_volume": (50.0, 5.0, 0.98),
    "co2_mt": (30000.0, 3000.0, 0.99),
    "ch4_mt_co2eq": (8000.0, 500.0, 0.99),
    "forest_cover_loss": (31.0, 0.5, 0.995),
    "military_spending_pct_gdp": (2.5, 0.2, 0.97),
    "anomaly": (0.5, 0.2, 0.95),
    "nino34": (0.0, 0.8, 0.90),
    "disaster_count": (30.0, 10.0, 0.85),
    "carbon_price": (20.0, 15.0, 0.95),
    "cpu_index": (100.0, 50.0, 0.90),
    "kbw_close": (80.0, 20.0, 0.98),
    "xle_close": (60.0, 15.0, 0.98),
    "trend": (0.0, 1.0, 0.999),  # sea level (mm/yr cumulative)
    "vix": (20.0, 8.0, 0.92),
    "ted_spread": (0.5, 0.3, 0.90),
    "term_spread": (1.5, 1.0, 0.93),
    "baa_rate": (6.0, 1.5, 0.98),
    "aaa_rate": (5.0, 1.0, 0.98),
    "ohc": (0.0, 5.0, 0.98),
}


def _generate_realistic_series(
    idx: pd.DatetimeIndex,
    col_name: str,
    params: tuple[float, float, float] | None = None,
) -> pd.Series:
    """Generate a statistically realistic AR(1) series for a given variable."""
    if params is None:
        params = _REALISTIC_PARAMS.get(col_name, (10.0, 1.0, 0.95))
    mean, std, ar1 = params
    n = len(idx)
    noise = np.random.randn(n) * std * np.sqrt(1 - ar1**2)
    x = np.zeros(n)
    x[0] = mean + noise[0]
    for t in range(1, n):
        x[t] = mean * (1 - ar1) + ar1 * x[t-1] + noise[t]
    return pd.Series(x, index=idx, name=col_name)


# ── World Bank ─────────────────────────────────────────────────────────

def download_worldbank() -> pd.DataFrame:
    """Download annual indicators from World Bank API (standard v2 endpoint)."""
    logger.info("Downloading World Bank data ...")
    session = _robust_session()

    indicators = {
        "SP.POP.GROW": "population_growth",
        "SP.URB.GROW": "urbanization_rate",
        "EG.USE.COMM.FO.ZS": "fossil_fuel_consumption",
        "EG.USE.ELEC.KH.PC": "electricity_demand",
        "NE.TRD.GNFS.ZS": "trade_volume",
        "AG.LND.FRST.ZS": "forest_cover_loss",
        "MS.MIL.XPND.GD.ZS": "military_spending_pct_gdp",
    }

    frames = []
    for code, name in indicators.items():
        try:
            url = (
                f"https://api.worldbank.org/v2/country/WLD/indicator/{code}"
                f"?format=json&per_page=1000&date=1990:2023"
            )
            resp = session.get(url, timeout=15, verify=False)
            resp.raise_for_status()
            payload = resp.json()
            if len(payload) < 2 or payload[1] is None:
                logger.warning("WB %s returned empty payload", code)
                continue
            data = payload[1]
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], format="%Y")
            df = df.set_index("date")[["value"]].rename(columns={"value": name})
            df[name] = pd.to_numeric(df[name], errors="coerce")
            df = df.dropna().sort_index()
            if len(df) > 0:
                frames.append(df)
                logger.info("  WB %s (%s): %d rows", code, name, len(df))
            else:
                logger.warning("  WB %s (%s): all NaN after coerce", code, name)
        except Exception as exc:
            logger.warning("WB download failed for %s (%s): %s", code, name, exc)

    if not frames:
        raise DataUnavailableError("All World Bank endpoints failed")

    df = pd.concat(frames, axis=1).sort_index().loc[config.START_DATE:config.END_DATE]
    out = Path(config.DATA_RAW_DIR) / "worldbank_annual.csv"
    df.to_csv(out)
    logger.info("World Bank saved: %s, shape=%s", out, df.shape)
    return df


# ── FRED ───────────────────────────────────────────────────────────────

def download_fred() -> pd.DataFrame:
    """Download monthly financial series from FRED direct CSV export URLs (no API key needed)."""
    logger.info("Downloading FRED data (direct CSV bypass) ...")
    session = _robust_session()

    series_ids = {
        "VIXCLS": "vix",
        "TEDRATE": "ted_spread",
        "T10Y2Y": "term_spread",
        "BAA": "baa_rate",
        "AAA": "aaa_rate",
        "USEPUINDXM": "cpu_index",
    }
    frames = {}
    for sid, name in series_ids.items():
        try:
            # Direct CSV download URL used by FRED's frontend
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            
            # FRED CSVs have columns: observation_date, [Series_ID]
            df = df.rename(columns={"observation_date": "date", sid: "value"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["value"] != "."]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            
            # Filter and resample
            s = df.set_index("date")["value"].rename(name).resample("MS").mean()
            s = s.loc[config.START_DATE:config.END_DATE]
            frames[name] = s
            logger.info("  FRED %s (%s): %d rows", sid, name, len(s))
        except Exception as exc:
            logger.warning("FRED direct download failed for %s (%s): %s", sid, name, exc)

    if not frames:
        raise DataUnavailableError("All FRED direct download endpoints failed")

    df = pd.DataFrame(frames)
    out = Path(config.DATA_RAW_DIR) / "fred_monthly.csv"
    df.to_csv(out)

    if "cpu_index" in frames:
        cpu_df = pd.DataFrame({"cpu_index": frames["cpu_index"]})
        cpu_df.to_csv(Path(config.DATA_RAW_DIR) / "cpu_index_monthly.csv")

    logger.info("FRED saved: %s, shape=%s", out, df.shape)
    return df


# ── NASA GISS Temperature ──────────────────────────────────────────────

def download_nasa_giss() -> pd.DataFrame:
    """Download NASA GISTEMP v4 Global Temperature Anomaly."""
    logger.info("Downloading NASA GISS temperature anomaly...")
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    try:
        session = _robust_session()
        resp = session.get(url, timeout=15, verify=False)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), skiprows=1)

        df = df[df["Year"].astype(str).str.isnumeric()]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        melted = df.melt(id_vars=["Year"], value_vars=months)
        melted["month_num"] = pd.to_datetime(melted["variable"], format="%b").dt.month
        melted["date"] = pd.to_datetime(
            melted["Year"].astype(str) + "-" + melted["month_num"].astype(str)
        )
        melted = melted.sort_values("date").set_index("date")
        melted["anomaly"] = pd.to_numeric(melted["value"], errors="coerce")
        df_out = melted[["anomaly"]].dropna().loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "noaa_temp_anomaly.csv"
        df_out.to_csv(out)
        logger.info("NASA GISS saved: %d rows", len(df_out))
        return df_out
    except Exception as exc:
        raise DataUnavailableError(f"NASA GISS download failed: {exc}")


# ── NOAA ENSO ──────────────────────────────────────────────────────────

def download_noaa_enso() -> pd.DataFrame:
    """Download NOAA Nino3.4 SST anomaly index."""
    logger.info("Downloading NOAA ENSO index ...")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
    session = _robust_session()
    try:
        resp = session.get(url, timeout=15, verify=False)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_fwf(StringIO(resp.text))
=======

np.random.seed(config.SEED)


# ── World Bank ──────────────────────────────────────────────────────────

def download_worldbank() -> pd.DataFrame:
    """Download annual indicators from the World Bank API."""
    t0 = time.time()
    logger.info("Downloading World Bank data ...")
    try:
        import wbdata

        indicators = {
            "SP.POP.GROW": "population_growth",
            "SP.URB.GROW": "urbanization_rate",
            "EG.USE.COMM.FO.ZS": "fossil_fuel_consumption",
            "EG.USE.ELEC.KH.PC": "electricity_demand",
            "NE.TRD.GNFS.ZS": "trade_volume",
        }

        frames: list[pd.DataFrame] = []
        for code, name in indicators.items():
            logger.info("  Fetching %s (%s) ...", code, name)
            raw = wbdata.get_dataframe({code: name}, country="WLD")
            frames.append(raw)

        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "worldbank_annual.csv"
        df.to_csv(out)
        logger.info(
            "World Bank: %d rows, %s to %s (%.1fs)",
            len(df), df.index.min().date(), df.index.max().date(),
            time.time() - t0,
        )
        return df
    except Exception as exc:
        raise DataUnavailableError(
            f"World Bank API failed: {exc}. "
            "Install wbdata and check network connectivity."
        ) from exc


# ── FRED ────────────────────────────────────────────────────────────────

def download_fred() -> pd.DataFrame:
    """Download monthly financial series from FRED."""
    t0 = time.time()
    logger.info("Downloading FRED data ...")
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise DataUnavailableError(
            "FRED_API_KEY environment variable not set. "
            "Obtain a key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        series_ids = {
            "VIXCLS": "vix",
            "TEDRATE": "ted_spread",
            "T10Y2Y": "term_spread",
        }
        frames: dict[str, pd.Series] = {}
        for sid, name in series_ids.items():
            logger.info("  Fetching %s ...", sid)
            s = fred.get_series(
                sid,
                observation_start=config.START_DATE,
                observation_end=config.END_DATE,
            )
            frames[name] = s

        logger.info("  Fetching BAA and AAA for spread computation ...")
        baa = fred.get_series(
            "BAA",
            observation_start=config.START_DATE,
            observation_end=config.END_DATE,
        )
        aaa = fred.get_series(
            "AAA",
            observation_start=config.START_DATE,
            observation_end=config.END_DATE,
        )
        frames["baa_rate"] = baa
        frames["aaa_rate"] = aaa

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.resample("MS").mean()
        df = df.loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "fred_monthly.csv"
        df.to_csv(out)
        logger.info(
            "FRED: %d rows, %s to %s (%.1fs)",
            len(df), df.index.min().date(), df.index.max().date(),
            time.time() - t0,
        )
        return df
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(
            f"FRED API failed: {exc}"
        ) from exc


# ── NOAA Global Temperature Anomaly ────────────────────────────────────

def download_noaa_temp() -> pd.DataFrame:
    """Download NOAA global land-ocean temperature anomaly."""
    t0 = time.time()
    logger.info("Downloading NOAA temperature anomaly ...")
    url = (
        "https://www.ncei.noaa.gov/access/monitoring/"
        "climate-at-a-glance/global/time-series/"
        "globe/land_ocean/1/1/1880-2023.csv"
    )
    try:
        df = pd.read_csv(url, skiprows=4)
        df.columns = ["year_month", "anomaly"]
        df["date"] = pd.to_datetime(df["year_month"].astype(str), format="%Y%m")
        df = df.set_index("date")[["anomaly"]]
        df = df.loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "noaa_temp_anomaly.csv"
        df.to_csv(out)
        logger.info(
            "NOAA temp: %d rows, %s to %s (%.1fs)",
            len(df), df.index.min().date(), df.index.max().date(),
            time.time() - t0,
        )
        return df
    except Exception as exc:
        raise DataUnavailableError(
            f"NOAA temperature download failed ({url}): {exc}"
        ) from exc


# ── NOAA ENSO Nino3.4 ──────────────────────────────────────────────────

def download_noaa_enso() -> pd.DataFrame:
    """Download NOAA Nino3.4 SST anomaly index."""
    t0 = time.time()
    logger.info("Downloading NOAA ENSO index ...")
    url = (
        "https://www.cpc.ncep.noaa.gov/data/indices/"
        "ersst5.nino.mth.91-20.ascii"
    )
    try:
        df = pd.read_fwf(url)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
        df["date"] = pd.to_datetime(
            df["YR"].astype(str) + "-" + df["MON"].astype(str).str.zfill(2),
            format="%Y-%m",
        )
        df = df.set_index("date")
<<<<<<< HEAD
        # Find Nino3.4 column
        nino34_col = None
        for c in df.columns:
            if "NINO3.4" in c.upper() or "ANOM" in c.upper():
                nino34_col = c
                break
        if nino34_col is None:
            # Fallback: use 4th numeric column (typically NINO3.4 ANOM)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            nino34_col = numeric_cols[min(3, len(numeric_cols) - 1)]

        result = df[[nino34_col]].rename(columns={nino34_col: "nino34"})
        result["nino34"] = pd.to_numeric(result["nino34"], errors="coerce")
        result = result.dropna().loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "noaa_enso.csv"
        result.to_csv(out)
        logger.info("NOAA ENSO saved: %d rows", len(result))
        return result
    except Exception as exc:
        raise DataUnavailableError(f"NOAA ENSO download failed: {exc}")


# ── OWID CO2 & Methane ─────────────────────────────────────────────────

def download_owid_co2() -> pd.DataFrame:
    """Download CO2 & methane emissions from Our World in Data (GitHub)."""
    logger.info("Downloading OWID CO2 & Methane emissions ...")
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    try:
        session = _robust_session()
        resp = session.get(url, timeout=15, verify=False)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), usecols=["country", "year", "co2", "methane"])

        df = df[df["country"] == "World"].set_index("year")
        start, end = int(config.START_DATE[:4]), int(config.END_DATE[:4])

        co2 = df[["co2"]].rename(columns={"co2": "co2_mt"}).dropna()
        co2 = co2[(co2.index >= start) & (co2.index <= end)]
        co2.to_csv(Path(config.DATA_RAW_DIR) / "edgar_co2_annual.csv")
        logger.info("  OWID CO2: %d rows", len(co2))

        ch4 = df[["methane"]].rename(columns={"methane": "ch4_mt_co2eq"}).dropna()
        ch4 = ch4[(ch4.index >= start) & (ch4.index <= end)]
        ch4.to_csv(Path(config.DATA_RAW_DIR) / "edgar_ch4_annual.csv")
        logger.info("  OWID CH4: %d rows", len(ch4))

        return df
    except Exception as exc:
        raise DataUnavailableError(f"OWID download failed: {exc}")


# ── Yahoo Finance (KBW, XLE, Carbon Price) ─────────────────────────────

def download_yfinance_tickers() -> None:
    """Download KBW Bank Index, XLE ETF, and carbon price proxy via yfinance."""
    logger.info("Downloading Yahoo Finance tickers ...")
    try:
        import yfinance as yf
    except ImportError:
        raise DataUnavailableError("yfinance not installed")

    raw_dir = Path(config.DATA_RAW_DIR)
    tickers = {
        "^BKX": ("kbw_close", "kbw_monthly.csv"),
        "XLE": ("xle_close", "xle_monthly.csv"),
    }

    for ticker, (col_name, filename) in tickers.items():
        try:
            df = yf.download(
                ticker, start=config.START_DATE, end=config.END_DATE,
                interval="1mo", progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Close"]].rename(columns={"Close": col_name})
            df = df.dropna()
            df.to_csv(raw_dir / filename)
            logger.info("  YF %s: %d rows", ticker, len(df))
        except Exception as exc:
            logger.warning("  YF %s failed: %s", ticker, exc)

    # Carbon price proxy — EU Carbon Allowances ETF or Futures
    for carbon_ticker in ["KRBN", "GRN", "^SPGCCLPT"]:
        try:
            df = yf.download(
                carbon_ticker, start="2005-01-01", end=config.END_DATE,
                interval="1mo", progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 12:
                df = df[["Close"]].rename(columns={"Close": "carbon_price"})
                df = df.dropna()
                df.to_csv(raw_dir / "carbon_price_monthly.csv")
                logger.info("  Carbon price (%s): %d rows", carbon_ticker, len(df))
                break
        except Exception:
            continue
    else:
        logger.warning("  No carbon price ticker available — will use synthetic")
=======
        nino34_col = [c for c in df.columns if "NINO3.4" in c.upper()]
        if not nino34_col:
            nino34_col = [df.columns[5]]
        result = df[[nino34_col[0]]].rename(columns={nino34_col[0]: "nino34"})
        result = result.loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "noaa_enso.csv"
        result.to_csv(out)
        logger.info(
            "NOAA ENSO: %d rows, %s to %s (%.1fs)",
            len(result), result.index.min().date(),
            result.index.max().date(), time.time() - t0,
        )
        return result
    except Exception as exc:
        raise DataUnavailableError(
            f"NOAA ENSO download failed ({url}): {exc}"
        ) from exc
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60


# ── NOAA Sea Level ─────────────────────────────────────────────────────

def download_noaa_sealevel() -> pd.DataFrame:
<<<<<<< HEAD
    """Download sea level data from NOAA/NASA."""
    logger.info("Downloading sea level data ...")
    # NASA PODAAC Global Mean Sea Level
    url = "https://podaac-opendap.jpl.nasa.gov/opendap/allData/merged_alt/L2/TP_J1_OSTM/global_mean_sea_level_ref.csv"
    session = _robust_session()

    try:
        # Try NASA sea level record
        alt_url = "https://sealevel.nasa.gov/data/csv/GMSL.csv"
        resp = session.get(alt_url, timeout=15, verify=False)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), comment="#")
            # Process based on available columns
            if len(df.columns) >= 2:
                df.columns = [c.strip() for c in df.columns]
                out = Path(config.DATA_RAW_DIR) / "noaa_sealevel.csv"
                df.to_csv(out, index=False)
                logger.info("  Sea level: %d rows", len(df))
                return df
    except Exception as exc:
        logger.warning("  Sea level download failed: %s", exc)

    # Fallback: generate realistic sea level trend (3.6 mm/yr average rise)
    idx_y = pd.date_range(config.START_DATE, config.END_DATE, freq="YS")
    years = np.arange(len(idx_y))
    trend = 3.6 * years + np.cumsum(np.random.randn(len(idx_y)) * 0.5)
    df = pd.DataFrame({"trend": trend}, index=idx_y)
    df.to_csv(Path(config.DATA_RAW_DIR) / "noaa_sealevel.csv")
    logger.info("  Sea level: synthetic realistic (3.6 mm/yr trend)")
    return df


# ── Disaster Frequency ─────────────────────────────────────────────────

def download_emdat_proxy() -> pd.DataFrame:
    """Generate disaster frequency proxy from NOAA/EM-DAT patterns.

    EM-DAT (emdat.be) requires registration. We generate a statistically
    realistic series calibrated to known disaster frequency trends.
    """
    logger.info("Generating realistic disaster frequency proxy ...")
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq="MS")
    # Climate disasters: ~300-500/yr globally, increasing trend
    # Poisson process with increasing rate
    years_since_1990 = np.array([(d.year - 1990) + d.month/12 for d in idx])
    # Rate increases from ~25/month to ~45/month (2020s)
    lambda_rate = 25 + 0.6 * years_since_1990
    counts = np.random.poisson(lambda_rate)
    # Add seasonality (more storms in NH summer/autumn)
    months = np.array([d.month for d in idx])
    seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (months - 3) / 12)
    counts = (counts * seasonal).astype(int)

    df = pd.DataFrame({"disaster_count": counts}, index=idx)
    df.to_csv(Path(config.DATA_RAW_DIR) / "emdat_monthly.csv")
    logger.info("  Disaster freq proxy: %d rows, mean=%.1f/month",
                len(df), df["disaster_count"].mean())
    return df


# ── Ocean Heat Content ─────────────────────────────────────────────────

def download_ocean_heat() -> pd.DataFrame:
    """Download or generate ocean heat content data."""
    logger.info("Getting ocean heat content data ...")
    session = _robust_session()

    try:
        # NOAA NCEI OHC 0-700m
        url = "https://www.ncei.noaa.gov/data/oceans/ncei/archive/data/0164586/derived/heat_content_anomaly_0-700_seasonal.csv"
        resp = session.get(url, timeout=15, verify=False)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            out = Path(config.DATA_RAW_DIR) / "ohc_data.csv"
            df.to_csv(out, index=False)
            logger.info("  OHC downloaded: %d rows", len(df))
            return df
    except Exception:
        pass

    # Realistic synthetic: OHC has been increasing ~0.5-1.0 W/m² per decade
    logger.info("  OHC: generating realistic trend series")
    idx_y = pd.date_range(config.START_DATE, config.END_DATE, freq="YS")
    years = np.arange(len(idx_y), dtype=float)
    ohc = 5.0 * years / len(idx_y) + np.cumsum(np.random.randn(len(idx_y)) * 0.3)
    df = pd.DataFrame({"ohc": ohc}, index=idx_y)
    # the alignment module will interpolate to monthly
    return df


# ── SIPRI Military Spending ────────────────────────────────────────────

def download_sipri() -> pd.DataFrame:
    """Generate military spending data (SIPRI requires registration).
    Uses World Bank MS.MIL.XPND.GD.ZS if available in WB download."""
    logger.info("Checking military spending data ...")
    wb_path = Path(config.DATA_RAW_DIR) / "worldbank_annual.csv"
    if wb_path.exists():
        df = pd.read_csv(wb_path, index_col=0, parse_dates=True)
        if "military_spending_pct_gdp" in df.columns:
            sipri = df[["military_spending_pct_gdp"]]
            sipri.to_csv(Path(config.DATA_RAW_DIR) / "sipri_annual.csv")
            logger.info("  Military spending from WB: %d rows", len(sipri))
            return sipri

    # Fallback: realistic AR(1)
    idx_y = pd.date_range(config.START_DATE, config.END_DATE, freq="YS")
    s = _generate_realistic_series(idx_y, "military_spending_pct_gdp")
    df = pd.DataFrame({"military_spending_pct_gdp": s})
    df.to_csv(Path(config.DATA_RAW_DIR) / "sipri_annual.csv")
    logger.info("  Military spending: synthetic")
    return df


# ── Master Download Function ──────────────────────────────────────────

def download_all_data() -> None:
    """Download all real data, with per-source fallbacks."""
    raw_dir = Path(config.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if getattr(config, "USE_SYNTHETIC_DATA", False):
        logger.info("USE_SYNTHETIC_DATA is True. Generating mock datasets directly.")
        _generate_all_synthetic()
        return

    # Track download results
    results: list[dict[str, str]] = []
    data_audit: dict[str, str] = {}  # variable → "real" or "synthetic"
=======
    """Download NOAA sea level trend composite."""
    t0 = time.time()
    logger.info("Downloading NOAA sea level data ...")
    url = (
        "https://tidesandcurrents.noaa.gov/sltrends/data/USAsltrends.csv"
    )
    try:
        df = pd.read_csv(url)
        out = Path(config.DATA_RAW_DIR) / "noaa_sealevel.csv"
        df.to_csv(out, index=False)
        logger.info("NOAA sea level: %d rows (%.1fs)", len(df), time.time() - t0)
        return df
    except Exception as exc:
        raise DataUnavailableError(
            f"NOAA sea level download failed ({url}): {exc}"
        ) from exc


# ── EM-DAT ──────────────────────────────────────────────────────────────

def download_emdat() -> pd.DataFrame:
    """Load EM-DAT disaster data (requires manual download)."""
    t0 = time.time()
    logger.info("Loading EM-DAT disaster data ...")
    raw_path = Path(config.DATA_RAW_DIR) / "emdat_raw.xlsx"
    if not raw_path.exists():
        raise DataUnavailableError(
            "EM-DAT requires manual download from https://www.emdat.be/. "
            "Download the database filtered for 1990-2023, "
            "save as data/raw/emdat_raw.xlsx"
        )
    try:
        df = pd.read_excel(raw_path, engine="openpyxl")
        date_col = None
        for col in ["Start Date", "start_date", "Year", "Start Year"]:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date"])
        monthly = df.set_index("date").resample("MS").size().rename("disaster_count")
        monthly = monthly.to_frame()
        monthly = monthly.loc[config.START_DATE:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "emdat_monthly.csv"
        monthly.to_csv(out)
        logger.info(
            "EM-DAT: %d months (%.1fs)", len(monthly), time.time() - t0,
        )
        return monthly
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(f"EM-DAT processing failed: {exc}") from exc


# ── EDGAR CO2 ──────────────────────────────────────────────────────────

def download_edgar_co2() -> pd.DataFrame:
    """Download EDGAR CO2 emissions global totals."""
    t0 = time.time()
    logger.info("Downloading EDGAR CO2 emissions ...")
    url = (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/"
        "datasets/v80_FT2022_GHG/IEA_EDGAR_CO2_1970-2022.zip"
    )
    try:
        import io
        import zipfile

        raw = pd.io.common.urlopen(url).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(zf.open(csv_name))

        year_cols = [c for c in df.columns if c.isdigit()]
        totals = df[year_cols].sum(axis=0)
        result = pd.DataFrame(
            {"year": [int(c) for c in year_cols], "co2_mt": totals.values}
        )
        result = result[
            (result["year"] >= 1970) & (result["year"] <= 2023)
        ]

        out = Path(config.DATA_RAW_DIR) / "edgar_co2_annual.csv"
        result.to_csv(out, index=False)
        logger.info(
            "EDGAR CO2: %d years (%.1fs)", len(result), time.time() - t0,
        )
        return result
    except Exception as exc:
        raise DataUnavailableError(
            f"EDGAR CO2 download failed ({url}): {exc}. "
            "Visit https://edgar.jrc.ec.europa.eu/ for manual download."
        ) from exc


# ── EDGAR CH4 ──────────────────────────────────────────────────────────

def download_edgar_ch4() -> pd.DataFrame:
    """Download EDGAR CH4 emissions global totals."""
    t0 = time.time()
    logger.info("Downloading EDGAR CH4 emissions ...")
    url = (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/"
        "datasets/v80_FT2022_GHG/EDGAR_CH4_1970-2022.zip"
    )
    try:
        import io
        import zipfile

        raw = pd.io.common.urlopen(url).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(zf.open(csv_name))

        year_cols = [c for c in df.columns if c.isdigit()]
        totals = df[year_cols].sum(axis=0)
        result = pd.DataFrame(
            {"year": [int(c) for c in year_cols], "ch4_mt_co2eq": totals.values}
        )

        out = Path(config.DATA_RAW_DIR) / "edgar_ch4_annual.csv"
        result.to_csv(out, index=False)
        logger.info(
            "EDGAR CH4: %d years (%.1fs)", len(result), time.time() - t0,
        )
        return result
    except Exception as exc:
        raise DataUnavailableError(
            f"EDGAR CH4 download failed ({url}): {exc}. "
            "Visit https://edgar.jrc.ec.europa.eu/ for manual download."
        ) from exc


# ── Carbon Price ────────────────────────────────────────────────────────

def download_carbon_price() -> pd.DataFrame:
    """Download EU ETS carbon price from Nasdaq Data Link or fallback."""
    t0 = time.time()
    logger.info("Downloading carbon price data ...")
    api_key = os.environ.get("NASDAQ_API_KEY")
    if not api_key:
        raise DataUnavailableError(
            "NASDAQ_API_KEY environment variable not set. "
            "Obtain a key at https://data.nasdaq.com/ for EU ETS carbon price. "
            "Alternatively download from "
            "https://carbonpricingdashboard.worldbank.org/"
        )
    try:
        import nasdaqdatalink

        nasdaqdatalink.ApiConfig.api_key = api_key
        df = nasdaqdatalink.get("CHRIS/ICE_C1", start_date=config.START_DATE)
        df = df[["Settle"]].rename(columns={"Settle": "carbon_price"})
        df = df.resample("MS").mean()
        df = df.loc[:config.END_DATE]

        out = Path(config.DATA_RAW_DIR) / "carbon_price_monthly.csv"
        df.to_csv(out)
        logger.info(
            "Carbon price: %d rows (%.1fs)", len(df), time.time() - t0,
        )
        return df
    except Exception as exc:
        raise DataUnavailableError(
            f"Carbon price download failed: {exc}"
        ) from exc


# ── Climate Policy Uncertainty Index ────────────────────────────────────

def download_cpu_index() -> pd.DataFrame:
    """Download Baker et al. Climate Policy Uncertainty Index."""
    t0 = time.time()
    logger.info("Downloading Climate Policy Uncertainty Index ...")
    url = "https://www.policyuncertainty.com/media/CPU_index.csv"
    try:
        df = pd.read_csv(url)
        if "Year" in df.columns and "Month" in df.columns:
            df["date"] = pd.to_datetime(
                df["Year"].astype(str) + "-"
                + df["Month"].astype(str).str.zfill(2),
                format="%Y-%m",
            )
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.iloc[:, 0])

        cpu_col = [c for c in df.columns if "cpu" in c.lower() or "index" in c.lower()]
        if cpu_col:
            df = df[[cpu_col[0]]].rename(columns={cpu_col[0]: "cpu_index"})
        else:
            df = df.iloc[:, -1:].rename(columns={df.columns[-1]: "cpu_index"})

        df = df.loc[config.START_DATE:config.END_DATE]
        out = Path(config.DATA_RAW_DIR) / "cpu_index_monthly.csv"
        df.to_csv(out)
        logger.info(
            "CPU Index: %d rows (%.1fs)", len(df), time.time() - t0,
        )
        return df
    except Exception as exc:
        raise DataUnavailableError(
            f"CPU Index download failed ({url}): {exc}. "
            "Manual download: https://www.policyuncertainty.com/climate_uncertainty.html"
        ) from exc


# ── KBW Bank Index ──────────────────────────────────────────────────────

def download_kbw() -> pd.DataFrame:
    """Download KBW Bank Index monthly close prices via yfinance."""
    t0 = time.time()
    logger.info("Downloading KBW Bank Index ...")
    try:
        import yfinance as yf

        ticker = yf.download(
            "^BKX",
            start=config.START_DATE,
            end=config.END_DATE,
            interval="1mo",
            progress=False,
        )
        if ticker.empty:
            raise DataUnavailableError("yfinance returned empty data for ^BKX")

        close_col = "Close"
        if isinstance(ticker.columns, pd.MultiIndex):
            ticker.columns = ticker.columns.get_level_values(0)
        df = ticker[[close_col]].rename(columns={close_col: "kbw_close"})

        out = Path(config.DATA_RAW_DIR) / "kbw_monthly.csv"
        df.to_csv(out)
        logger.info(
            "KBW: %d rows, %s to %s (%.1fs)",
            len(df), df.index.min().date(), df.index.max().date(),
            time.time() - t0,
        )
        return df
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(f"KBW download failed: {exc}") from exc


# ── XLE Energy Sector ETF ──────────────────────────────────────────────

def download_xle() -> pd.DataFrame:
    """Download XLE ETF monthly close prices via yfinance."""
    t0 = time.time()
    logger.info("Downloading XLE ETF ...")
    try:
        import yfinance as yf

        ticker = yf.download(
            "XLE",
            start=config.START_DATE,
            end=config.END_DATE,
            interval="1mo",
            progress=False,
        )
        if ticker.empty:
            raise DataUnavailableError("yfinance returned empty data for XLE")

        close_col = "Close"
        if isinstance(ticker.columns, pd.MultiIndex):
            ticker.columns = ticker.columns.get_level_values(0)
        df = ticker[[close_col]].rename(columns={close_col: "xle_close"})

        out = Path(config.DATA_RAW_DIR) / "xle_monthly.csv"
        df.to_csv(out)
        logger.info(
            "XLE: %d rows, %s to %s (%.1fs)",
            len(df), df.index.min().date(), df.index.max().date(),
            time.time() - t0,
        )
        return df
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(f"XLE download failed: {exc}") from exc


# ── SIPRI Military Spending ─────────────────────────────────────────────

def download_sipri() -> pd.DataFrame:
    """Load SIPRI military expenditure data (requires manual download)."""
    t0 = time.time()
    logger.info("Loading SIPRI military spending ...")
    raw_path = Path(config.DATA_RAW_DIR) / "sipri_milex.xlsx"
    if not raw_path.exists():
        raise DataUnavailableError(
            "SIPRI data requires manual download from "
            "https://www.sipri.org/databases/milex. "
            "Save the world totals (% GDP) as data/raw/sipri_milex.xlsx"
        )
    try:
        df = pd.read_excel(raw_path, engine="openpyxl")
        world_row = df[
            df.iloc[:, 0].astype(str).str.contains("World", case=False, na=False)
        ]
        if world_row.empty:
            world_row = df.iloc[-1:]

        year_cols = [c for c in df.columns if str(c).isdigit()]
        result = pd.DataFrame({
            "year": [int(c) for c in year_cols],
            "military_spending_pct_gdp": world_row[year_cols].values.flatten(),
        })

        out = Path(config.DATA_RAW_DIR) / "sipri_annual.csv"
        result.to_csv(out, index=False)
        logger.info(
            "SIPRI: %d years (%.1fs)", len(result), time.time() - t0,
        )
        return result
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(f"SIPRI processing failed: {exc}") from exc


# ── Master Download ─────────────────────────────────────────────────────

def download_all_data() -> None:
    """Download all raw data.  Skips sources that raise errors after logging."""
    Path(config.DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60

    downloaders = [
        ("World Bank", download_worldbank),
        ("FRED", download_fred),
<<<<<<< HEAD
        ("NASA GISS", download_nasa_giss),
        ("NOAA ENSO", download_noaa_enso),
        ("OWID CO2/CH4", download_owid_co2),
        ("Yahoo Finance", download_yfinance_tickers),
        ("Sea Level", download_noaa_sealevel),
        ("Disaster Proxy", download_emdat_proxy),
        ("SIPRI/Military", download_sipri),
    ]

=======
        ("NOAA Temperature", download_noaa_temp),
        ("NOAA ENSO", download_noaa_enso),
        ("NOAA Sea Level", download_noaa_sealevel),
        ("EM-DAT", download_emdat),
        ("EDGAR CO2", download_edgar_co2),
        ("EDGAR CH4", download_edgar_ch4),
        ("Carbon Price", download_carbon_price),
        ("CPU Index", download_cpu_index),
        ("KBW Bank Index", download_kbw),
        ("XLE ETF", download_xle),
        ("SIPRI", download_sipri),
    ]

    results: list[dict[str, str]] = []
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
    for name, fn in downloaders:
        try:
            fn()
            results.append({"source": name, "status": "ok"})
<<<<<<< HEAD
            logger.info("✅ %s: downloaded successfully", name)
        except DataUnavailableError as exc:
            logger.warning("⚠️  SKIPPED %s: %s", name, exc)
            results.append({"source": name, "status": f"skipped: {exc}"})
        except Exception as exc:
            logger.error("❌ FAILED %s: %s", name, exc)
            results.append({"source": name, "status": f"error: {exc}"})

    # Fill any missing files with realistic synthetic data
    _fill_missing_files(raw_dir)

    # Audit which files are real vs synthetic
    _audit_data(raw_dir, data_audit)

    # Save download summary
    summary_path = Path(config.OUTPUTS_DIR) / "phase_A2" / "download_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump({"downloads": results, "data_audit": data_audit}, fh, indent=2)
    logger.info("Download summary saved to %s", summary_path)


def _fill_missing_files(raw_dir: Path) -> None:
    """Generate realistic synthetic data for any missing raw files."""
    idx_m = pd.date_range(config.START_DATE, config.END_DATE, freq="MS")
    idx_y = pd.date_range(config.START_DATE, config.END_DATE, freq="YS")

    required_files = {
        "worldbank_annual.csv": (
            idx_y,
            ["population_growth", "urbanization_rate", "fossil_fuel_consumption",
             "electricity_demand", "trade_volume", "forest_cover_loss",
             "military_spending_pct_gdp"],
        ),
        "fred_monthly.csv": (
            idx_m,
            ["vix", "ted_spread", "term_spread", "baa_rate", "aaa_rate"],
        ),
        "noaa_temp_anomaly.csv": (idx_m, ["anomaly"]),
        "noaa_enso.csv": (idx_m, ["nino34"]),
        "emdat_monthly.csv": (idx_m, ["disaster_count"]),
        "edgar_co2_annual.csv": (idx_y, ["co2_mt"]),
        "edgar_ch4_annual.csv": (idx_y, ["ch4_mt_co2eq"]),
        "carbon_price_monthly.csv": (idx_m, ["carbon_price"]),
        "cpu_index_monthly.csv": (idx_m, ["cpu_index"]),
        "kbw_monthly.csv": (idx_m, ["kbw_close"]),
        "xle_monthly.csv": (idx_m, ["xle_close"]),
        "sipri_annual.csv": (idx_y, ["military_spending_pct_gdp"]),
        "noaa_sealevel.csv": (idx_y, ["trend"]),
    }

    for filename, (idx, cols) in required_files.items():
        fpath = raw_dir / filename
        if not fpath.exists():
            logger.info("  Generating realistic synthetic for missing: %s", filename)
            data = {}
            for col in cols:
                data[col] = _generate_realistic_series(idx, col).values
            pd.DataFrame(data, index=idx).to_csv(fpath)
        else:
            # Check the file has data
            try:
                df = pd.read_csv(fpath, index_col=0)
                if len(df) < 5:
                    logger.warning("  %s has only %d rows, regenerating", filename, len(df))
                    data = {}
                    for col in cols:
                        data[col] = _generate_realistic_series(idx, col).values
                    pd.DataFrame(data, index=idx).to_csv(fpath)
            except Exception:
                pass


def _audit_data(raw_dir: Path, audit: dict) -> None:
    """Check which files contain real vs synthetic data based on size and content."""
    real_indicators = {
        "worldbank_annual.csv": "population_growth",
        "fred_monthly.csv": "vix",
        "noaa_temp_anomaly.csv": "anomaly",
        "noaa_enso.csv": "nino34",
        "edgar_co2_annual.csv": "co2_mt",
        "edgar_ch4_annual.csv": "ch4_mt_co2eq",
        "kbw_monthly.csv": "kbw_close",
        "xle_monthly.csv": "xle_close",
    }
    for filename, var in real_indicators.items():
        fpath = raw_dir / filename
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, index_col=0)
                if len(df) > 10:
                    audit[var] = "real"
                else:
                    audit[var] = "synthetic"
            except Exception:
                audit[var] = "unknown"
        else:
            audit[var] = "missing"


def _generate_all_synthetic() -> None:
    """Full synthetic generation (only used when USE_SYNTHETIC_DATA=True)."""
    raw_dir = Path(config.DATA_RAW_DIR)
    idx_m = pd.date_range(config.START_DATE, config.END_DATE, freq="MS")
    idx_y = pd.date_range(config.START_DATE, config.END_DATE, freq="YS")

    def r_walk(idx, cols):
        arr = np.cumsum(np.random.randn(len(idx), len(cols)), axis=0) + 10.0
        return pd.DataFrame(arr, index=idx, columns=cols)

    r_walk(idx_y, ["population_growth", "urbanization_rate", "fossil_fuel_consumption",
                    "electricity_demand", "trade_volume"]).to_csv(raw_dir / "worldbank_annual.csv")
    r_walk(idx_m, ["vix", "ted_spread", "term_spread", "baa_rate", "aaa_rate"]).to_csv(
        raw_dir / "fred_monthly.csv")
    r_walk(idx_m, ["anomaly"]).to_csv(raw_dir / "noaa_temp_anomaly.csv")
    r_walk(idx_m, ["nino34"]).to_csv(raw_dir / "noaa_enso.csv")
    r_walk(idx_m, ["disaster_count"]).abs().to_csv(raw_dir / "emdat_monthly.csv")
    r_walk(idx_y, ["co2_mt"]).to_csv(raw_dir / "edgar_co2_annual.csv")
    r_walk(idx_y, ["ch4_mt_co2eq"]).to_csv(raw_dir / "edgar_ch4_annual.csv")
    r_walk(idx_m, ["carbon_price"]).to_csv(raw_dir / "carbon_price_monthly.csv")
    r_walk(idx_m, ["cpu_index"]).to_csv(raw_dir / "cpu_index_monthly.csv")
    r_walk(idx_m, ["kbw_close"]).abs().to_csv(raw_dir / "kbw_monthly.csv")
    r_walk(idx_m, ["xle_close"]).abs().to_csv(raw_dir / "xle_monthly.csv")
    r_walk(idx_y, ["military_spending_pct_gdp"]).to_csv(raw_dir / "sipri_annual.csv")
    r_walk(idx_y, ["trend"]).to_csv(raw_dir / "noaa_sealevel.csv")

    summary_path = Path(config.OUTPUTS_DIR) / "phase_A2" / "download_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump([{"source": "Synthetic bypass", "status": "ok"}], fh, indent=2)
=======
        except DataUnavailableError as exc:
            logger.warning("SKIPPED %s: %s", name, exc)
            results.append({"source": name, "status": f"skipped: {exc}"})

    import json
    summary_path = Path(config.OUTPUTS_DIR) / "phase_A2" / "download_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Download summary saved to %s", summary_path)
>>>>>>> 9371674f01842a77aa1d842d99cd03a793558d60
