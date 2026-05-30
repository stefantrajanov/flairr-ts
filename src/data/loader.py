"""
FLAIRR-TS – OWID Energy Dataset Loader
Downloads the CSV from OWID's public endpoint if not already present locally,
then loads it into a validated, typed DataFrame with column renaming to match
the FLAIRR-TS pipeline convention (country → location-free; we keep 'country').
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

OWID_CSV_URL = "https://owid-public.owid.io/data/energy/owid-energy-data.csv"

# Columns the codebook defines as mandatory
REQUIRED_COLUMNS = {"country", "year"}

# Indicators available for forecasting (subset from codebook)
FORECASTABLE_INDICATORS = [
    "electricity_demand",
    "primary_energy_consumption",
    "fossil_share_elec",
    "renewables_share_elec",
    "low_carbon_share_elec",
    "solar_share_elec",
    "wind_share_elec",
    "coal_share_elec",
    "gas_share_elec",
    "nuclear_share_elec",
    "electricity_generation",
    "fossil_electricity",
    "renewables_electricity",
    "carbon_intensity_elec",
]


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_owid_energy(
    data_path: Optional[str | Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load the OWID Energy dataset into a DataFrame.

    Resolution order:
    1. ``data_path`` argument (if given and file exists).
    2. ``OWID_DATA_PATH`` env-var (if set and file exists).
    3. Auto-download from OWID's public CDN, saving to the default path.

    Parameters
    ----------
    data_path:
        Explicit local path to ``owid-energy-data.csv``.
    force_download:
        If True, re-download even when a local copy exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with a MultiIndex-friendly structure indexed by
        (country, year), dtypes inferred by pandas.
    """
    resolved_path = _resolve_path(data_path)

    if force_download or not resolved_path.exists():
        logger.info("Downloading OWID Energy dataset from %s …", OWID_CSV_URL)
        _download(OWID_CSV_URL, resolved_path)

    logger.info("Loading OWID Energy data from %s", resolved_path)
    df = pd.read_csv(resolved_path, low_memory=False)
    df = _validate_and_clean(df)
    return df


def _resolve_path(data_path: Optional[str | Path]) -> Path:
    """Return the resolved local CSV path, checking env-var as fallback."""
    if data_path is not None:
        return Path(data_path)

    env_path = os.getenv("OWID_DATA_PATH")
    if env_path:
        return Path(env_path)

    # Default: place next to the energy-data-master directory
    return Path(__file__).parent.parent.parent / "energy-data-master" / "owid-energy-data.csv"


def _download(url: str, dest: Path) -> None:
    """Stream-download ``url`` to ``dest``, creating parent directories."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    logger.info("Downloaded %s → %s (%.1f MB)", url, dest, dest.stat().st_size / 1e6)


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist, coerce 'year' to int, and sort.
    The OWID CSV uses 'country' (not 'location') – we keep this unchanged.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"OWID CSV missing required columns: {missing}")

    df["year"] = df["year"].astype(int)
    df.sort_values(["country", "year"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def list_countries(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique country / region names."""
    return sorted(df["country"].unique().tolist())


def list_indicators(df: pd.DataFrame) -> list[str]:
    """Return forecastable indicator columns that actually exist in this CSV."""
    return [col for col in FORECASTABLE_INDICATORS if col in df.columns]


def get_country_series(
    df: pd.DataFrame,
    country: str,
    indicator: str,
) -> pd.DataFrame:
    """
    Extract a single country's time series for one indicator.

    Returns a DataFrame with columns ['year', indicator], sorted ascending,
    with years that are *entirely* NaN for that indicator dropped from the
    leading/trailing edges only (internal NaNs preserved for interpolation).
    """
    if indicator not in df.columns:
        raise ValueError(f"Indicator '{indicator}' not found in dataset. Available: {list_indicators(df)}")

    sub = df[df["country"] == country][["year", indicator]].copy()
    if sub.empty:
        raise ValueError(f"Country '{country}' not found in dataset.")

    sub.sort_values("year", inplace=True)
    sub.reset_index(drop=True, inplace=True)

    # Trim leading / trailing all-NaN rows
    first_valid = sub[indicator].first_valid_index()
    last_valid = sub[indicator].last_valid_index()
    if first_valid is None:
        raise ValueError(f"Indicator '{indicator}' is entirely NaN for '{country}'.")

    sub = sub.loc[first_valid:last_valid].reset_index(drop=True)
    return sub
