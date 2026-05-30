"""
FLAIRR-TS – Forecaster Agent
LangChain chain backed by a Claude model (default: claude-haiku-4-5).
Ingests:
  • Dynamically refined instructions
  • Token-efficient historical context string
  • M retrieved analogue segments (few-shot RAG context)
Emits a structured ForecastOutput with H numerical predictions.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────────────

class ForecastOutput(BaseModel):
    """Structured output expected from the Forecaster Agent."""

    predictions: List[float] = Field(
        description=(
            "An ordered list of exactly H floating-point predictions for the "
            "next H years. No explanations – numbers only."
        )
    )
    reasoning: str = Field(
        description=(
            "Concise step-by-step reasoning: identify trend direction, rate of "
            "change, inflection points, and how the retrieved analogues informed "
            "the forecast."
        )
    )
    certainty: int = Field(
        ge=0, le=100,
        description="Confidence score 0–100 for the overall forecast quality.",
    )
    certainty_reasoning: str = Field(
        description="One-sentence explanation of the certainty score.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────

_FORECASTER_SYSTEM = (
    "You are a precise quantitative time-series analyst. "
    "Your only job is to predict future numerical values for an energy indicator. "
    "Base your predictions strictly on the historical data and the analogous "
    "historical scenarios provided. "
    "Output ONLY valid numbers – never placeholder text like <value>."
)

_FORECASTER_USER = """\
## Task
Forecast the {indicator} for **{country}** for the years **{forecast_years}**.

## Variable context
- Indicator : {indicator}
- Unit      : {unit}
- Country   : {country}
- Context   : last {L} years  →  predict next {H} years

## Forecasting instructions (refined by the Refiner Agent)
{instructions}

## Retrieved historical analogues  (Pearson-correlated similar trajectories)
{retrieved_context}

## Target historical sequence  (your primary input — extend this)
{historical_data}

## Output requirements
- Produce exactly {H} predictions matching years {forecast_years}.
- Each prediction must be a concrete floating-point number — no ranges, no nulls.
- Maintain numerical scale consistency with the historical sequence.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Chain factory
# ──────────────────────────────────────────────────────────────────────────────

def get_forecaster_chain(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> Runnable:
    """
    Build the Forecaster LangChain pipeline.

    Parameters
    ----------
    model_name  : Anthropic model ID. Falls back to ``FLAIRR_FORECASTER_MODEL``
                  env-var, then ``claude-haiku-4-5``.
    temperature : Sampling temperature (0 = deterministic).

    Returns
    -------
    A LangChain ``Runnable`` that accepts a dict with keys matching the
    prompt template and returns a ``ForecastOutput`` instance.
    """
    resolved_model = (
        model_name
        or os.getenv("FLAIRR_FORECASTER_MODEL", "claude-haiku-4-5")
    )
    logger.info("Forecaster using model: %s", resolved_model)

    llm = ChatAnthropic(
        model=resolved_model,
        temperature=temperature,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _FORECASTER_SYSTEM),
        ("human", _FORECASTER_USER),
    ])

    return prompt | llm.with_structured_output(ForecastOutput)


# ──────────────────────────────────────────────────────────────────────────────
# Resilient invoke wrapper
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def invoke_forecaster(
    chain: Runnable,
    country: str,
    indicator: str,
    unit: str,
    L: int,
    H: int,
    forecast_years: List[int],
    instructions: str,
    retrieved_context: str,
    historical_data: str,
) -> ForecastOutput:
    """
    Invoke the Forecaster chain with automatic retry on transient errors.
    Validates that the returned predictions list has exactly H elements;
    applies a regex fallback parser if structured output fails.
    """
    try:
        result: ForecastOutput = chain.invoke({
            "country": country,
            "indicator": indicator,
            "unit": unit,
            "L": L,
            "H": H,
            "forecast_years": ", ".join(str(y) for y in forecast_years),
            "instructions": instructions,
            "retrieved_context": retrieved_context,
            "historical_data": historical_data,
        })

        if len(result.predictions) != H:
            logger.warning(
                "Forecaster returned %d predictions, expected %d. Truncating/padding.",
                len(result.predictions), H,
            )
            result.predictions = _fix_prediction_length(result.predictions, H)

        return result

    except Exception as exc:
        logger.error("Forecaster chain failed: %s. Will retry.", exc)
        raise


def _fix_prediction_length(predictions: List[float], H: int) -> List[float]:
    """Truncate or forward-pad a predictions list to length H."""
    if len(predictions) >= H:
        return predictions[:H]
    # Pad by repeating last value
    last = predictions[-1] if predictions else 0.0
    return predictions + [last] * (H - len(predictions))


# ──────────────────────────────────────────────────────────────────────────────
# Regex fallback parser (used when structured output breaks)
# ──────────────────────────────────────────────────────────────────────────────

def parse_predictions_from_text(text: str, H: int) -> List[float]:
    """
    Extract up to H float values from a free-text LLM response.
    Searches for numeric patterns; returns what it finds or raises ValueError.
    """
    # Match sequences of numbers (int or float) separated by commas/whitespace
    pattern = r"[-+]?\d+(?:\.\d+)?"
    matches = re.findall(pattern, text)
    numbers = [float(m) for m in matches]

    if not numbers:
        raise ValueError(f"No numeric predictions found in Forecaster output:\n{text}")

    return _fix_prediction_length(numbers, H)
