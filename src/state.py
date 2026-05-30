"""
FLAIRR-TS State Model
Centralised Pydantic schema that flows through the entire agent pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────

class RetrievedSegment(BaseModel):
    """One analogue sequence returned by the Retrieval Agent."""

    location: str = Field(description="Country / region that produced this segment.")
    start_year: int = Field(description="First year of the context window slice.")
    end_year: int = Field(description="Last year of the context window slice.")
    context_years: List[int] = Field(description="Ordered list of years in the context window.")
    context_values: List[float] = Field(description="Ordered indicator values for context_years (length L).")
    lookahead_years: List[int] = Field(description="Ordered list of years in the look-ahead window.")
    lookahead_values: List[float] = Field(description="Actual ground-truth values for the H look-ahead years.")
    correlation: float = Field(description="Pearson r between this segment and the target context window.")


class PromptEvaluation(BaseModel):
    """Record of a single forecasting attempt and its quality score."""

    iteration: int = Field(description="1-indexed iteration number.")
    instructions: str = Field(description="Full instruction text sent to the Forecaster.")
    raw_predictions: List[float] = Field(description="Raw float predictions from the Forecaster.")
    mae: float = Field(description="Mean Absolute Error against the evaluation ground truth.")
    refiner_learnings: Optional[str] = Field(
        default=None,
        description="Learnings produced by the Refiner after this iteration.",
    )
    done_signal: bool = Field(
        default=False,
        description="Whether the Refiner emitted a done_signal for this iteration.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Central state
# ──────────────────────────────────────────────────────────────────────────────

class FLAIRRState(BaseModel):
    """
    Central state object threaded through the FLAIRR-TS pipeline.

    Immutability note: This model is *not* frozen – the orchestrator mutates
    fields in-place during the refinement loop.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    location: str = Field(description="Target country / region name (matches 'country' column).")
    target_indicator: str = Field(description="OWID column name being forecast.")
    indicator_unit: str = Field(default="", description="Human-readable unit string (e.g. 'TWh').")

    # ── Temporal configuration ────────────────────────────────────────────────
    y_current: int = Field(description="First forecast year (exclusive end of context window).")
    L: int = Field(description="Context window length in years.")
    H: int = Field(description="Forecast horizon in years.")

    # ── Core data ─────────────────────────────────────────────────────────────
    context_years: List[int] = Field(default_factory=list, description="Years in the context window [y_current-L .. y_current-1].")
    current_context: List[float] = Field(description="Indicator values for context_years (length L).")
    eval_years: List[int] = Field(default_factory=list, description="Years in the evaluation window [y_current .. y_current+H-1].")
    eval_ground_truth: List[float] = Field(description="Actual indicator values for eval_years (length H).")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_segments: List[RetrievedSegment] = Field(default_factory=list)

    # ── Prompt refinement history ─────────────────────────────────────────────
    prompt_history: List[PromptEvaluation] = Field(default_factory=list)
    base_instructions: str = Field(
        default="Forecast the next values by carefully extrapolating the recent historical trend.",
        description="Seed instructions used at iteration 1.",
    )
    current_instructions: str = Field(default="")

    # ── Best checkpoint ───────────────────────────────────────────────────────
    best_instructions: Optional[str] = Field(default=None)
    best_forecast: List[float] = Field(default_factory=list)
    min_mae: float = Field(default=float("inf"))

    # ── Flow control ──────────────────────────────────────────────────────────
    iteration: int = Field(default=0)
    max_iterations: int = Field(default=5)
    stopping_threshold: float = Field(default=0.05, description="τ_stop: min relative MAE improvement to continue.")
    early_stop: bool = Field(default=False)

    # ── Metadata ──────────────────────────────────────────────────────────────
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def record_evaluation(
        self,
        instructions: str,
        predictions: List[float],
        mae: float,
        learnings: Optional[str] = None,
        done_signal: bool = False,
    ) -> None:
        """Append a completed evaluation to prompt_history and update best checkpoint."""
        record = PromptEvaluation(
            iteration=self.iteration,
            instructions=instructions,
            raw_predictions=predictions,
            mae=mae,
            refiner_learnings=learnings,
            done_signal=done_signal,
        )
        self.prompt_history.append(record)

        if mae < self.min_mae:
            self.min_mae = mae
            self.best_instructions = instructions
            self.best_forecast = list(predictions)

    def relative_improvement(self) -> float:
        """Relative MAE improvement from the previous to the current iteration."""
        if len(self.prompt_history) < 2:
            return 1.0
        prev_mae = self.prompt_history[-2].mae
        curr_mae = self.prompt_history[-1].mae
        if prev_mae == 0:
            return 0.0
        return (prev_mae - curr_mae) / prev_mae
