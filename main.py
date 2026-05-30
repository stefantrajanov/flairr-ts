"""
FLAIRR-TS – CLI Entry Point
Usage examples:

  # Run with uv (recommended):
  uv run main.py --country Germany --indicator electricity_demand --y-current 2018

  # With custom hyperparameters:
  uv run main.py \
      --country "United States" \
      --indicator primary_energy_consumption \
      --y-current 2015 \
      --L 12 --H 5 --M 3 \
      --max-iter 7 \
      --forecaster-model claude-haiku-4-5 \
      --refiner-model claude-sonnet-4-5

  # List available countries / indicators:
  uv run main.py --list-countries
  uv run main.py --list-indicators
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports (keep startup fast for --help / --list-* flags)
# ──────────────────────────────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    """Execute the full FLAIRR-TS pipeline and print results."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from src.data.loader import list_countries, list_indicators, load_owid_energy
    from src.orchestrator import run_flairr_ts

    console = Console()

    # ── Special list modes ─────────────────────────────────────────────────
    if args.list_countries or args.list_indicators:
        console.print("[bold]Loading dataset …[/bold]")
        df = load_owid_energy(args.data_path)
        if args.list_countries:
            countries = list_countries(df)
            console.print(f"\n[bold green]{len(countries)} countries / regions:[/bold green]")
            for c in countries:
                console.print(f"  {c}")
        if args.list_indicators:
            indicators = list_indicators(df)
            console.print(f"\n[bold green]{len(indicators)} forecastable indicators:[/bold green]")
            for ind in indicators:
                console.print(f"  {ind}")
        return

    # ── Validate required args ─────────────────────────────────────────────
    for required in ("country", "indicator", "y_current"):
        if getattr(args, required, None) is None:
            console.print(f"[bold red]Error:[/bold red] --{required.replace('_', '-')} is required.")
            sys.exit(1)

    console.rule(f"[bold blue]FLAIRR-TS[/bold blue]")
    console.print(
        f"[bold]Country:[/bold] {args.country}  "
        f"[bold]Indicator:[/bold] {args.indicator}  "
        f"[bold]Y_current:[/bold] {args.y_current}"
    )

    # ── Load data once (pass to run_flairr_ts to avoid double-load) ────────
    df = load_owid_energy(args.data_path)

    # ── Run pipeline ───────────────────────────────────────────────────────
    result = run_flairr_ts(
        country=args.country,
        indicator=args.indicator,
        y_current=args.y_current,
        L=args.L,
        H=args.H,
        M=args.M,
        max_iterations=args.max_iter,
        tau_stop=args.tau_stop,
        forecaster_model=args.forecaster_model or None,
        refiner_model=args.refiner_model or None,
        df=df,
        verbose=True,
    )

    # ── Results table ──────────────────────────────────────────────────────
    console.rule("[bold green]Results[/bold green]")

    console.print(Panel(
        f"[bold]Best MAE:[/bold] {result['min_mae']:.4f}\n"
        f"[bold]Iterations run:[/bold] {result['state'].iteration}\n"
        f"[bold]Early stop:[/bold] {result['state'].early_stop}",
        title="Summary",
    ))

    tbl = Table(title="Evaluation Forecast vs Ground Truth")
    tbl.add_column("Year", style="cyan")
    tbl.add_column("Ground Truth", style="green")
    tbl.add_column("Best Forecast", style="yellow")
    tbl.add_column("Abs Error", style="red")

    for yr, gt, pred in zip(
        result["eval_years"],
        result["eval_ground_truth"],
        result["best_forecast"],
    ):
        tbl.add_row(
            str(yr),
            f"{gt:.2f}",
            f"{pred:.2f}",
            f"{abs(gt - pred):.2f}",
        )
    console.print(tbl)

    console.print("\n[bold]Best Forecasting Instructions:[/bold]")
    console.print(Panel(result["best_instructions"] or "N/A", border_style="blue"))

    # ── Iteration history ──────────────────────────────────────────────────
    hist_tbl = Table(title="Iteration History")
    hist_tbl.add_column("Iter", style="cyan")
    hist_tbl.add_column("MAE", style="yellow")
    hist_tbl.add_column("Done?", style="green")
    for rec in result["history"]:
        hist_tbl.add_row(
            str(rec.iteration),
            f"{rec.mae:.4f}",
            "✓" if rec.done_signal else "",
        )
    console.print(hist_tbl)

    # ── Optional JSON output ───────────────────────────────────────────────
    if args.output_json:
        out_path = Path(args.output_json)
        payload = {
            "country": args.country,
            "indicator": args.indicator,
            "y_current": args.y_current,
            "min_mae": result["min_mae"],
            "best_forecast": result["best_forecast"],
            "best_instructions": result["best_instructions"],
            "eval_years": result["eval_years"],
            "eval_ground_truth": result["eval_ground_truth"],
            "iterations": [
                {
                    "iteration": r.iteration,
                    "mae": r.mae,
                    "predictions": r.raw_predictions,
                    "instructions": r.instructions,
                    "learnings": r.refiner_learnings,
                    "done_signal": r.done_signal,
                }
                for r in result["history"]
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        console.print(f"\n[bold green]JSON results saved to:[/bold green] {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI definition
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flairr-ts",
        description="FLAIRR-TS: LLM-Agent Iterative Prompt Refinement for Time-Series Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Target ─────────────────────────────────────────────────────────────
    p.add_argument("--country",    type=str, help="Target country name (e.g. 'Germany').")
    p.add_argument("--indicator",  type=str, help="OWID column to forecast (e.g. 'electricity_demand').")
    p.add_argument("--y-current",  type=int, dest="y_current",
                   help="First year of the forecast / evaluation window.")

    # ── Hyperparameters ────────────────────────────────────────────────────
    p.add_argument("--L",         type=int,   default=None, help="Context window length (years). Default: 10.")
    p.add_argument("--H",         type=int,   default=None, help="Forecast horizon (years). Default: 3.")
    p.add_argument("--M",         type=int,   default=None, help="Number of retrieved analogues. Default: 2.")
    p.add_argument("--max-iter",  type=int,   default=None, dest="max_iter",
                   help="Max refinement iterations. Default: 5.")
    p.add_argument("--tau-stop",  type=float, default=None, dest="tau_stop",
                   help="Relative MAE improvement threshold for early stop. Default: 0.05.")

    # ── Model selection ────────────────────────────────────────────────────
    p.add_argument("--forecaster-model", type=str, default=None, dest="forecaster_model",
                   help="Anthropic model for the Forecaster agent.")
    p.add_argument("--refiner-model",    type=str, default=None, dest="refiner_model",
                   help="Anthropic model for the Refiner agent.")

    # ── Data ───────────────────────────────────────────────────────────────
    p.add_argument("--data-path", type=str, default=None, dest="data_path",
                   help="Path to owid-energy-data.csv (auto-downloaded if absent).")

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--output-json", type=str, default=None, dest="output_json",
                   help="Save full results as JSON to this path.")

    # ── Utility ────────────────────────────────────────────────────────────
    p.add_argument("--list-countries",  action="store_true", dest="list_countries",
                   help="Print all available countries and exit.")
    p.add_argument("--list-indicators", action="store_true", dest="list_indicators",
                   help="Print all forecastable indicators and exit.")

    return p


def cli_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    cli_main()
