"""
FLAIRR-TS – Main Orchestrator
Implements the full iterative refinement loop described in the paper:

  1. Partition OWID data into context / eval / historical-DB windows.
  2. Run the deterministic Retrieval Agent (Pearson correlation).
  3. Iterate for up to N iterations:
        a. Forecaster Agent  → predictions
        b. MAE evaluation    → update best checkpoint
        c. Refiner Agent     → improved instructions or done_signal
  4. Return best instructions + forecast + full audit trail.

All hyperparameters are read from environment variables with sensible defaults,
and can be overridden by passing kwargs to ``run_flairr_ts``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from src.agents.forecaster import ForecastOutput, get_forecaster_chain, invoke_forecaster
from src.agents.refiner import RefinerOutput, get_refiner_chain, invoke_refiner
from src.agents.retrieval import retrieve_similar_segments
from src.data.loader import load_owid_energy
from src.data.preprocessor import (
    build_historical_database,
    format_history_log,
    format_retrieved_segments,
    format_series,
    partition_series,
)
from src.state import FLAIRRState, RetrievedSegment

load_dotenv()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Seed instructions (used at iteration 1)
# ──────────────────────────────────────────────────────────────────────────────

_SEED_INSTRUCTIONS = (
    "Carefully extrapolate the recent historical trend. "
    "Use the retrieved analogues as secondary evidence for the direction and "
    "magnitude of change. Maintain numerical scale and unit consistency with "
    "the input data."
)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_flairr_ts(
    country: str,
    indicator: str,
    y_current: int,
    *,
    # Hyperparameters (fall back to env-vars then hard defaults)
    L: Optional[int] = None,
    H: Optional[int] = None,
    M: Optional[int] = None,
    max_iterations: Optional[int] = None,
    tau_stop: Optional[float] = None,
    # Model selection
    forecaster_model: Optional[str] = None,
    refiner_model: Optional[str] = None,
    # Data
    df=None,
    data_path: Optional[str] = None,
    # Misc
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full FLAIRR-TS iterative refinement pipeline.

    Parameters
    ----------
    country         : Target country name (as it appears in the OWID dataset).
    indicator       : OWID column to forecast (e.g. 'electricity_demand').
    y_current       : First year of the forecast / evaluation window.
    L               : Context window length in years (default: env FLAIRR_CONTEXT_LENGTH or 10).
    H               : Forecast horizon in years (default: env FLAIRR_HORIZON or 3).
    M               : Number of retrieved analogues (default: env FLAIRR_TOP_M or 2).
    max_iterations  : Max refinement iterations (default: env FLAIRR_MAX_ITERATIONS or 5).
    tau_stop        : Relative MAE improvement threshold (default: env FLAIRR_STOP_THRESHOLD or 0.05).
    forecaster_model: Override Forecaster model name.
    refiner_model   : Override Refiner model name.
    df              : Pre-loaded OWID DataFrame (skips file I/O).
    data_path       : Path to owid-energy-data.csv (auto-downloaded if absent).
    verbose         : Enable Rich progress logging.

    Returns
    -------
    dict with keys:
        state           : Final ``FLAIRRState`` object.
        best_forecast   : List[float] of H predictions from best iteration.
        best_instructions: str – prompt that achieved min_mae.
        min_mae         : float – best validation MAE achieved.
        history         : List of PromptEvaluation records.
    """
    # ── Resolve hyperparameters ────────────────────────────────────────────
    L            = L            or int(os.getenv("FLAIRR_CONTEXT_LENGTH", 10))
    H            = H            or int(os.getenv("FLAIRR_HORIZON", 3))
    M            = M            or int(os.getenv("FLAIRR_TOP_M", 2))
    max_iter     = max_iterations or int(os.getenv("FLAIRR_MAX_ITERATIONS", 5))
    tau          = tau_stop     or float(os.getenv("FLAIRR_STOP_THRESHOLD", 0.05))

    if verbose:
        _setup_logging()
    logger.info(
        "FLAIRR-TS | country=%s | indicator=%s | y_current=%d | L=%d H=%d M=%d",
        country, indicator, y_current, L, H, M,
    )

    # ── 1. Load data ───────────────────────────────────────────────────────
    if df is None:
        df = load_owid_energy(data_path)

    # Infer unit from codebook (optional, used for display only)
    unit = _infer_unit(indicator)

    # ── 2. Partition series ────────────────────────────────────────────────
    ctx_years, ctx_values, eval_years, eval_values = partition_series(
        df, country, indicator, y_current, L, H
    )
    logger.info(
        "Context: %d–%d | Eval: %d–%d",
        ctx_years[0], ctx_years[-1], eval_years[0], eval_years[-1],
    )

    # ── 3. Build historical DB & run retrieval ─────────────────────────────
    logger.info("Building historical candidate database …")
    candidates = build_historical_database(df, country, indicator, y_current, L, H)

    logger.info("Running Retrieval Agent (Pearson correlation) …")
    retrieved: List[RetrievedSegment] = retrieve_similar_segments(
        candidates, ctx_values, L, M
    )

    # Pre-render the retrieved context string once (static across all iterations)
    retrieved_dicts = [seg.model_dump() for seg in retrieved]
    raft_context_str = format_retrieved_segments(retrieved_dicts, unit)
    historical_data_str = format_series(ctx_years, ctx_values, unit)
    forecast_years = eval_years  # we predict the eval window for validation

    # ── 4. Initialise state ────────────────────────────────────────────────
    state = FLAIRRState(
        location=country,
        target_indicator=indicator,
        indicator_unit=unit,
        y_current=y_current,
        L=L, H=H,
        context_years=ctx_years,
        current_context=ctx_values,
        eval_years=eval_years,
        eval_ground_truth=eval_values,
        retrieved_segments=retrieved,
        base_instructions=_SEED_INSTRUCTIONS,
        current_instructions=_SEED_INSTRUCTIONS,
        max_iterations=max_iter,
        stopping_threshold=tau,
    )

    # ── 5. Build chains ────────────────────────────────────────────────────
    forecaster_chain = get_forecaster_chain(model_name=forecaster_model)
    refiner_chain    = get_refiner_chain(model_name=refiner_model)

    # ── 6. Iterative refinement loop ───────────────────────────────────────
    for k in range(1, max_iter + 1):
        state.iteration = k
        logger.info("── Iteration %d / %d ──────────────────────────", k, max_iter)

        # 6a. Forecast
        forecast_out: ForecastOutput = invoke_forecaster(
            chain=forecaster_chain,
            country=country,
            indicator=indicator,
            unit=unit,
            L=L, H=H,
            forecast_years=forecast_years,
            instructions=state.current_instructions,
            retrieved_context=raft_context_str,
            historical_data=historical_data_str,
        )
        predictions = forecast_out.predictions

        # 6b. Evaluate – MAE
        mae = _compute_mae(eval_values, predictions)
        logger.info(
            "Iter %d | MAE=%.4f | certainty=%d%% | prev_best=%.4f",
            k, mae, forecast_out.certainty, state.min_mae,
        )

        # 6c. Update best checkpoint
        prev_mae = state.min_mae
        state.record_evaluation(
            instructions=state.current_instructions,
            predictions=predictions,
            mae=mae,
        )

        # 6d. Early stop: relative improvement check (skip on first iteration)
        if k > 1 and prev_mae != float("inf"):
            rel_improvement = (prev_mae - mae) / prev_mae if prev_mae > 0 else 0.0
            if rel_improvement < tau:
                logger.info(
                    "Early stop: rel_improvement=%.4f < tau=%.4f", rel_improvement, tau
                )
                state.early_stop = True

        # 6e. Refine (even on early stop iteration, we still log learnings)
        history_str = format_history_log(
            [
                {
                    "iteration": rec.iteration,
                    "instructions": rec.instructions,
                    "predictions": rec.raw_predictions,
                    "mae": rec.mae,
                }
                for rec in state.prompt_history
            ],
            unit=unit,
        )

        refiner_out: RefinerOutput = invoke_refiner(
            chain=refiner_chain,
            country=country,
            indicator=indicator,
            unit=unit,
            eval_ground_truth=eval_values,
            eval_years=eval_years,
            history_log=history_str,
            current_instructions=state.current_instructions,
        )

        # Annotate the last history entry with refiner learnings
        state.prompt_history[-1].refiner_learnings = refiner_out.learnings
        state.prompt_history[-1].done_signal = refiner_out.done_signal

        logger.info(
            "Refiner | done_signal=%s | confidence=%s",
            refiner_out.done_signal, refiner_out.confidence,
        )
        logger.debug("Refiner learnings: %s", refiner_out.learnings)

        # 6f. Check termination
        if state.early_stop or refiner_out.done_signal:
            logger.info("Terminating loop at iteration %d.", k)
            break

        # 6g. Update instructions for next iteration
        state.current_instructions = refiner_out.next_prompt

    # ── 7. Final output ────────────────────────────────────────────────────
    logger.info(
        "FLAIRR-TS complete | best_mae=%.4f | iterations=%d",
        state.min_mae, state.iteration,
    )

    return {
        "state": state,
        "best_forecast": state.best_forecast,
        "best_instructions": state.best_instructions,
        "min_mae": state.min_mae,
        "history": state.prompt_history,
        "retrieved_segments": retrieved,
        "context_years": ctx_years,
        "context_values": ctx_values,
        "eval_years": eval_years,
        "eval_ground_truth": eval_values,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_mae(ground_truth: List[float], predictions: List[float]) -> float:
    """
    MAE = (1/H) * Σ |x_i − x̂_i|

    Handles length mismatches by truncating to the shorter sequence.
    """
    n = min(len(ground_truth), len(predictions))
    if n == 0:
        return float("inf")
    gt = np.array(ground_truth[:n], dtype=float)
    pr = np.array(predictions[:n], dtype=float)
    return float(np.mean(np.abs(gt - pr)))


def _infer_unit(indicator: str) -> str:
    """Return a human-readable unit string based on the indicator name."""
    if indicator.endswith("_share_elec") or indicator.endswith("_share_energy"):
        return "%"
    if "per_capita" in indicator:
        return "kWh"
    if "per_gdp" in indicator:
        return "kWh/$"
    if "intensity" in indicator:
        return "gCO₂/kWh"
    if "emissions" in indicator:
        return "Mt CO₂"
    return "TWh"


def _setup_logging() -> None:
    """Configure Rich-style logging if not already set up."""
    if logging.getLogger().handlers:
        return
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        )
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
