"""
FLAIRR-TS – Refiner Agent
LangChain chain backed by Claude Sonnet (claude-sonnet-4-5 by default).
Acts as a meta-prompt optimizer: given the full iteration history
(prompt_i, forecast_i, MAE_i), it produces refined Forecaster instructions
and decides when to terminate the loop.
"""
from __future__ import annotations

import logging
import os
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

class RefinerOutput(BaseModel):
    """Structured output from the Refiner Agent."""

    learnings: str = Field(
        description=(
            "Analytical summary of observed error patterns: "
            "systematic over/under-estimation, missed trend inflections, "
            "scale drift, lag effects, or volatility mismatches."
        )
    )
    next_prompt: str = Field(
        description=(
            "1–3 highly specific, actionable instructions for the Forecaster. "
            "These REPLACE the previous instructions entirely. "
            "Must be concrete rules (e.g. 'Scale predictions by +8%% if the last "
            "3 values show consistent annual growth > 5%%.'). "
            "NO placeholders like <value> or <year>."
        )
    )
    done_signal: bool = Field(
        description=(
            "Set to True if: (a) MAE is already very low and further refinement "
            "is unlikely to improve it, or (b) MAE has plateaued across the last "
            "2+ iterations with < 5%% relative improvement."
        )
    )
    confidence: str = Field(
        description="'High', 'Medium', or 'Low' confidence in the next_prompt effectiveness."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────

_REFINER_SYSTEM = """\
You are an expert Time-Series Forecasting Prompt Engineer and meta-optimizer.

Your role: Analyse the forecasting performance of a Forecaster Agent across \
multiple iterations and produce precise, actionable prompt improvements that \
will reduce Mean Absolute Error (MAE) on the next iteration.

Core principles:
1. DIAGNOSE before PRESCRIBING – identify the root cause of errors first.
2. Be SPECIFIC – vague instructions ('be more accurate') are useless.
3. QUANTIFY adjustments where possible ('increase predictions by ~5%').
4. If MAE has stopped improving, emit done_signal=True to prevent overfitting \
   the instructions to noise.
"""

_REFINER_USER = """\
## Forecasting Task Context
- Country   : {country}
- Indicator : {indicator}  ({unit})
- Evaluation ground truth : {eval_ground_truth}
  (These are the actual values for years {eval_years})

## Iteration History (chronological, oldest first)
{history_log}

## Currently active instructions (used in the last iteration)
{current_instructions}

## Your analysis tasks
1. Compare predictions vs ground truth for each iteration. Compute signed errors.
2. Look for patterns: consistent underestimation? Missed acceleration? Wrong scale?
3. Check whether the active instructions contributed to or corrected those errors.
4. Write 'learnings' summarising what you found.
5. Write 'next_prompt' with 1–3 precise corrective rules.
6. Set 'done_signal=True' if further refinement is unlikely to help.

Output only the structured JSON – no extra commentary.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Chain factory
# ──────────────────────────────────────────────────────────────────────────────

def get_refiner_chain(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
) -> Runnable:
    """
    Build the Refiner LangChain pipeline.

    Parameters
    ----------
    model_name  : Anthropic model ID. Falls back to ``FLAIRR_REFINER_MODEL``
                  env-var, then ``claude-sonnet-4-5``.
    temperature : Slight non-zero temp allows creative prompt generation.
    """
    resolved_model = (
        model_name
        or os.getenv("FLAIRR_REFINER_MODEL", "claude-sonnet-4-5")
    )
    logger.info("Refiner using model: %s", resolved_model)

    llm = ChatAnthropic(
        model=resolved_model,
        temperature=temperature,
        max_tokens=2048,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _REFINER_SYSTEM),
        ("human", _REFINER_USER),
    ])

    return prompt | llm.with_structured_output(RefinerOutput)


# ──────────────────────────────────────────────────────────────────────────────
# Resilient invoke wrapper
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def invoke_refiner(
    chain: Runnable,
    country: str,
    indicator: str,
    unit: str,
    eval_ground_truth: List[float],
    eval_years: List[int],
    history_log: str,
    current_instructions: str,
) -> RefinerOutput:
    """
    Invoke the Refiner with automatic retry on transient API errors.
    """
    gt_str = ", ".join(f"{y}:{round(v, 2)}" for y, v in zip(eval_years, eval_ground_truth))
    eval_years_str = ", ".join(str(y) for y in eval_years)

    result: RefinerOutput = chain.invoke({
        "country": country,
        "indicator": indicator,
        "unit": unit,
        "eval_ground_truth": gt_str,
        "eval_years": eval_years_str,
        "history_log": history_log,
        "current_instructions": current_instructions,
    })
    return result
