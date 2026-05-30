"""
FLAIRR-TS – Retrieval Agent (Deterministic)
Computes Pearson's r between the current context window and every historical
sliding window in the candidate pool, returning the top-M analogues with
their corresponding look-ahead values formatted for few-shot LLM prompting.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.state import RetrievedSegment

logger = logging.getLogger(__name__)


def retrieve_similar_segments(
    candidates: List[Dict[str, Any]],
    x_ctx: List[float],
    L: int,
    M: int = 2,
) -> List[RetrievedSegment]:
    """
    Select the top-M most correlated historical analogues.

    Algorithm
    ---------
    For each candidate window of length L in the pre-built historical
    database, compute the Pearson correlation coefficient between the
    target context vector and the candidate context vector.  Rank by
    |r| descending and return the top-M validated segments.

    Parameters
    ----------
    candidates : Output of ``build_historical_database`` – list of dicts
                 with keys: country, start_year, end_year,
                 context_years, context_values, lookahead_years, lookahead_values.
    x_ctx      : Target context window values (length L).
    L          : Expected context window length (used for validation).
    M          : Number of top analogues to return.

    Returns
    -------
    List of ``RetrievedSegment`` Pydantic objects, sorted by |r| descending.

    Notes
    -----
    - Windows with zero variance in either vector are skipped (correlation
      is undefined; such flat segments carry no trend information).
    - NaN correlations are silently discarded.
    """
    x_arr = np.array(x_ctx, dtype=float)

    if len(x_arr) != L:
        logger.warning("Context length mismatch: got %d, expected %d.", len(x_arr), L)
        return []

    if np.std(x_arr) == 0.0:
        logger.warning("Target context is constant – retrieval skipped.")
        return []

    scored: List[tuple[float, Dict[str, Any]]] = []

    for cand in candidates:
        c_arr = np.array(cand["context_values"], dtype=float)

        if len(c_arr) != L:
            continue
        if np.std(c_arr) == 0.0:
            continue
        if np.isnan(c_arr).any() or np.isnan(x_arr).any():
            continue

        corr_matrix = np.corrcoef(x_arr, c_arr)
        r = corr_matrix[0, 1]

        if np.isnan(r):
            continue

        scored.append((abs(r), r, cand))

    # Sort by absolute Pearson r descending
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:M]

    segments: List[RetrievedSegment] = []
    for abs_r, r, cand in top:
        seg = RetrievedSegment(
            location=cand["country"],
            start_year=cand["start_year"],
            end_year=cand["end_year"],
            context_years=cand["context_years"],
            context_values=cand["context_values"],
            lookahead_years=cand["lookahead_years"],
            lookahead_values=cand["lookahead_values"],
            correlation=round(r, 4),
        )
        segments.append(seg)
        logger.debug(
            "Retrieved analogue: %s [%d–%d] r=%.4f",
            cand["country"], cand["start_year"], cand["end_year"], r,
        )

    logger.info(
        "Retrieval: %d candidates scored, returning top %d (best r=%.4f)",
        len(scored),
        len(segments),
        scored[0][0] if scored else float("nan"),
    )
    return segments
