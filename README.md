# FLAIRR-TS

> **F**orecasting **L**LM-**A**gents with **I**terative **R**efinement and **R**etrieval for **T**ime **S**eries

Python implementation of the FLAIRR-TS framework (Jalori, Verma & Arik – EMNLP 2025), applied to the [Our World in Data Energy Dataset](https://github.com/owid/energy-data).

---

## How It Works

```
OWID Energy CSV
      │
      ▼
┌─────────────────────────┐
│  Data Partitioner       │  Slice → Context window (L yrs)
│  + Gap Filler           │          Eval window   (H yrs)
└────────────┬────────────┘          Historical DB (all prior)
             │
             ▼
┌─────────────────────────┐
│  Retrieval Agent        │  Pearson r → top-M analogues
│  (Deterministic)        │  (few-shot RAG context)
└────────────┬────────────┘
             │
    ┌────────▼────────┐
    │  k = 1 … N_iter │◄───────────────────────────┐
    └────────┬────────┘                             │
             │                                      │
             ▼                                      │
┌─────────────────────────┐                         │
│  Forecaster Agent       │  Claude Haiku            │
│  (LangChain + Claude)   │  → H predictions         │
└────────────┬────────────┘                         │
             │                                      │
             ▼                                      │
        MAE Evaluation                              │
        Best-checkpoint update                      │
             │                                      │
             ▼                                      │
┌─────────────────────────┐                         │
│  Refiner Agent          │  Claude Sonnet           │
│  (Meta-optimizer)       │  → refined instructions  │
└────────────┬────────────┘    or done_signal=True   │
             │                                      │
        done? ──No──────────────────────────────────┘
             │Yes
             ▼
     Best instructions + forecast returned
```

---

## Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone / enter the project
cd iterative-prompt-refinement-with-agents-for-time-series-data

# 3. Create virtual environment and install all dependencies
uv sync

# For development tools (pytest, ruff):
uv sync --extra dev

# 4. Configure environment
cp .env.example .env
# → Edit .env and set ANTHROPIC_API_KEY
```

The OWID dataset is **auto-downloaded** on first run. To use a local copy:
```bash
OWID_DATA_PATH=energy-data-master/owid-energy-data.csv
```

---

## Usage

```bash
# Discover available data
uv run main.py --list-countries
uv run main.py --list-indicators

# Run FLAIRR-TS on Germany's electricity demand
uv run main.py \
    --country Germany \
    --indicator electricity_demand \
    --y-current 2018

# Custom hyperparameters
uv run main.py \
    --country "United States" \
    --indicator primary_energy_consumption \
    --y-current 2015 \
    --L 12 --H 5 --M 3 \
    --max-iter 7

# Save results to JSON
uv run main.py \
    --country France \
    --indicator fossil_share_elec \
    --y-current 2019 \
    --output-json results/france_fossil.json
```

---

## Configuration

| Env variable              | Default                | Description                              |
|---------------------------|------------------------|------------------------------------------|
| `ANTHROPIC_API_KEY`       | *(required)*           | Anthropic API key                        |
| `FLAIRR_FORECASTER_MODEL` | `claude-haiku-4-5`     | Model for the Forecaster agent           |
| `FLAIRR_REFINER_MODEL`    | `claude-sonnet-4-5`    | Model for the Refiner agent              |
| `FLAIRR_CONTEXT_LENGTH`   | `10`                   | L – context window (years)               |
| `FLAIRR_HORIZON`          | `3`                    | H – forecast horizon (years)             |
| `FLAIRR_TOP_M`            | `2`                    | M – retrieved analogues                  |
| `FLAIRR_MAX_ITERATIONS`   | `5`                    | Max refinement iterations                |
| `FLAIRR_STOP_THRESHOLD`   | `0.05`                 | τ – relative MAE improvement threshold   |
| `OWID_DATA_PATH`          | *(auto-download)*      | Local path to owid-energy-data.csv       |

---

## Project Structure

```
.
├── pyproject.toml              ← uv / hatch project definition
├── .env.example                ← environment variable template
├── main.py                     ← CLI entry point  (uv run main.py)
├── src/
│   ├── __init__.py
│   ├── state.py                ← FLAIRRState Pydantic model
│   ├── orchestrator.py         ← Main refinement loop
│   ├── data/
│   │   ├── loader.py           ← OWID dataset loader + auto-download
│   │   └── preprocessor.py     ← Partitioning, gap-filling, serialisation
│   └── agents/
│       ├── retrieval.py        ← Pearson-correlation retrieval agent
│       ├── forecaster.py       ← LangChain + Claude Forecaster chain
│       └── refiner.py          ← LangChain + Claude Sonnet Refiner chain
├── tests/
│   ├── test_preprocessor.py
│   └── test_retrieval.py
└── energy-data-master/
    ├── owid-energy-codebook.csv
    └── owid-energy-data.csv    ← auto-downloaded on first run
```

---

## Running Tests

```bash
# Run all tests (no API keys needed)
uv run pytest

# With coverage report
uv run pytest --cov=src --cov-report=term-missing
```

---

## Mathematics

### Pearson Correlation (Retrieval)

$$r(X_{ctx}, W_i) = \frac{\sum_{t=1}^{L}(x_t - \bar{x})(w_t - \bar{w})}{\sqrt{\sum_{t=1}^{L}(x_t-\bar{x})^2 \cdot \sum_{t=1}^{L}(w_t-\bar{w})^2}}$$

### Mean Absolute Error (Evaluation)

$$MAE_k = \frac{1}{H}\sum_{i=1}^{H}|x_i - \hat{x}_i^{(k)}|$$

### Relative Improvement (Early Stop)

$$\Delta_k = \frac{MAE_{k-1} - MAE_k}{MAE_{k-1}} \quad \text{stop if } \Delta_k < \tau_{stop}$$

---

## Citation

```bibtex
@inproceedings{jalori2025flairrts,
  title     = {FLAIRR-TS – Forecasting LLM-Agents with Iterative Refinement and Retrieval for Time Series},
  author    = {Jalori, Gunjan and Verma, Preetika and Arik, Sercan O.},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-emnlp.834}
}
```
