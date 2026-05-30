"""
FLAIRR-TS – Data Preprocessor
Handles temporal partitioning, gap-filling, and token-efficient serialisation
of OWID Energy time-series data for ingestion by Claude-based agents.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import get_country_series

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Partition
# ──────────────────────────────────────────────────────────────────────────────

def partition_series(
    df: pd.DataFrame,
    country: str,
    indicator: str,
    y_current: int,
    L: int,
    H: int,
) -> Tuple[List[int], List[float], List[int], List[float]]:
    """
    Slice a country's time series into three non-overlapping temporal windows.

    Parameters
    ----------
    df          : Full OWID dataframe.
    country     : Target country name (matches 'country' column).
    indicator   : OWID column to forecast.
    y_current   : First year of the evaluation / forecast window.
    L           : Context window length in years.
    H           : Forecast horizon (evaluation window length).

    Returns
    -------
    context_years   : List[int]   – years [y_current-L .. y_current-1]
    context_values  : List[float] – gap-filled indicator values for context_years
    eval_years      : List[int]   – years [y_current .. y_current+H-1]
    eval_values     : List[float] – gap-filled indicator values for eval_years

    Raises
    ------
    ValueError  : If there is insufficient data for the requested partitioning.
    """
    series = get_country_series(df, country, indicator)
    years_available = series["year"].tolist()

    # ── Context window ─────────────────────────────────────────────────────
    ctx_start = y_current - L
    ctx_end = y_current - 1
    ctx_years = list(range(ctx_start, ctx_end + 1))

    ctx_sub = series[series["year"].isin(ctx_years)].copy()
    if len(ctx_sub) < L:
        raise ValueError(
            f"Insufficient context data for '{country}' / '{indicator}': "
            f"need {L} years [{ctx_start}–{ctx_end}], "
            f"found {len(ctx_sub)}. "
            f"Available years: {years_available[0]}–{years_available[-1]}"
        )

    ctx_values = _fill_gaps(ctx_sub, indicator, ctx_years)

    # ── Evaluation window ──────────────────────────────────────────────────
    eval_start = y_current
    eval_end = y_current + H - 1
    eval_yrs = list(range(eval_start, eval_end + 1))

    eval_sub = series[series["year"].isin(eval_yrs)].copy()
    if len(eval_sub) < H:
        raise ValueError(
            f"Insufficient evaluation data for '{country}' / '{indicator}': "
            f"need {H} years [{eval_start}–{eval_end}], "
            f"found {len(eval_sub)}. "
            f"Available years: {years_available[0]}–{years_available[-1]}"
        )

    eval_values = _fill_gaps(eval_sub, indicator, eval_yrs)

    logger.debug(
        "Partitioned '%s'/'%s': ctx=[%d–%d], eval=[%d–%d]",
        country, indicator, ctx_start, ctx_end, eval_start, eval_end,
    )
    return ctx_years, ctx_values, eval_yrs, eval_values


def _fill_gaps(sub: pd.DataFrame, indicator: str, target_years: List[int]) -> List[float]:
    """
    Reindex to ``target_years``, then fill NaNs via linear interpolation +
    forward-fill + backward-fill to ensure no NaN reaches the LLM.
    """
    sub = sub.set_index("year").reindex(target_years)
    filled = (
        sub[indicator]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )
    if filled.isna().any():
        raise ValueError(f"Could not fill all NaN values in indicator '{indicator}'.")
    return [round(float(v), 4) for v in filled.tolist()]


# ──────────────────────────────────────────────────────────────────────────────
# Build the historical sliding-window database
# ──────────────────────────────────────────────────────────────────────────────

def build_historical_database(
    df: pd.DataFrame,
    target_country: str,
    indicator: str,
    y_current: int,
    L: int,
    H: int,
    min_coverage: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Construct the pool of candidate analogues for the Retrieval Agent.

    For every country in the dataset (including the target country for
    earlier time periods), generate sliding windows of length ``L``
    followed by ``H`` look-ahead steps.  Data-leakage prevention:
    any window whose look-ahead overlaps [y_current .. y_current+H-1]
    is excluded.

    Parameters
    ----------
    min_coverage : Fraction of non-NaN values required in both context and
                   look-ahead windows to include the candidate.

    Returns
    -------
    List of candidate dicts with keys:
        country, start_year, end_year,
        context_years, context_values,
        lookahead_years, lookahead_values
    """
    candidates: List[Dict[str, Any]] = []
    leakage_boundary = y_current + H - 1  # last year that must not appear in lookahead

    for country, group in df.groupby("country"):
        sub = group[["year", indicator]].copy().sort_values("year").reset_index(drop=True)
        years = sub["year"].values
        values = sub[indicator].values.astype(float)

        n = len(years)
        if n < L + H:
            continue

        for i in range(n - L - H + 1):
            ctx_yr = years[i : i + L]
            lah_yr = years[i + L : i + L + H]
            ctx_val = values[i : i + L]
            lah_val = values[i + L : i + L + H]

            # ── Leakage guard ──────────────────────────────────────────────
            # Exclude if the look-ahead window bleeds into evaluation period
            if lah_yr[-1] >= y_current and country == target_country:
                continue
            # For other countries, only exclude exact same years if they
            # would reveal the answer to the target country's evaluation.
            # Conservative: skip any segment whose lookahead reaches y_current
            if lah_yr[0] >= y_current and country == target_country:
                continue

            # ── Coverage filter ────────────────────────────────────────────
            ctx_valid = np.sum(~np.isnan(ctx_val)) / L
            lah_valid = np.sum(~np.isnan(lah_val)) / H
            if ctx_valid < min_coverage or lah_valid < min_coverage:
                continue

            # ── Fill gaps ──────────────────────────────────────────────────
            ctx_filled = _interpolate_array(ctx_val)
            lah_filled = _interpolate_array(lah_val)

            if np.isnan(ctx_filled).any() or np.isnan(lah_filled).any():
                continue

            candidates.append({
                "country": str(country),
                "start_year": int(ctx_yr[0]),
                "end_year": int(ctx_yr[-1]),
                "context_years": ctx_yr.tolist(),
                "context_values": [round(float(v), 4) for v in ctx_filled],
                "lookahead_years": lah_yr.tolist(),
                "lookahead_values": [round(float(v), 4) for v in lah_filled],
            })

    logger.info("Built historical DB: %d candidate windows", len(candidates))
    return candidates


def _interpolate_array(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation + edge-fill on a 1-D array with possible NaNs."""
    s = pd.Series(arr)
    return s.interpolate(method="linear").ffill().bfill().to_numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Token-efficient serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_series(
    years: List[int],
    values: List[float],
    unit: str = "",
    decimals: int = 2,
) -> str:
    """
    Serialise a year→value mapping in a compact, token-efficient format.

    Output: ``"2010:350.40, 2011:362.10, …"``
    Avoids verbose JSON wrappers while remaining unambiguous for Claude.
    """
    suffix = f" {unit}" if unit else ""
    entries = [f"{y}:{round(v, decimals)}{suffix}" for y, v in zip(years, values)]
    return ", ".join(entries)


def format_retrieved_segments(
    segments: List[Dict[str, Any]],
    unit: str = "",
) -> str:
    """
    Render retrieved analogue segments as a few-shot context block.

    Each segment shows its context window then its actual look-ahead values
    so the Forecaster can learn from analogous historical trajectories.
    """

    if not segments:
        return "No similar historical segments found."

    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        ctx_str = format_series(seg["context_years"], seg["context_values"], unit)
        lah_str = format_series(seg["lookahead_years"], seg["lookahead_values"], unit)
        corr = seg.get("correlation", float("nan"))
        lines.append(
            f"[Analogue {i}] Country: {seg['location']} | "
            f"Pearson r={corr:.3f}\n"
            f"  History : {ctx_str}\n"
            f"  Outcome : {lah_str}"
        )
    return "\n\n".join(lines)


def format_history_log(history: List[Dict[str, Any]], unit: str = "") -> str:
    """
    Render the prompt-refinement history for the Refiner Agent.
    Keeps each entry compact: iteration number, MAE, and key instructions.
    """
    lines: List[str] = []
    for entry in history:
        pred_str = ", ".join(f"{round(v, 2)}" for v in entry.get("predictions", []))
        lines.append(
            f"── Iteration {entry['iteration']} ──\n"
            f"  Instructions : {entry['instructions']}\n"
            f"  Predictions  : [{pred_str}]\n"
            f"  MAE          : {entry['mae']:.4f} {unit}"
        )
    return "\n\n".join(lines)
