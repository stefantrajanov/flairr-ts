"""
Unit tests – src/data/preprocessor.py
Runs without network access or API keys.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    _fill_gaps,
    build_historical_database,
    format_history_log,
    format_retrieved_segments,
    format_series,
    partition_series,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_df():
    """Minimal two-country OWID-like DataFrame for deterministic testing."""
    rows = []
    for country in ["Germany", "France"]:
        for year in range(2000, 2025):
            val = 300.0 + (year - 2000) * 5.0 + (0 if country == "Germany" else 50.0)
            rows.append({"country": country, "year": year, "electricity_demand": val})
    df = pd.DataFrame(rows)
    return df


@pytest.fixture()
def df_with_nans(simple_df):
    """DataFrame with deliberate internal NaN values."""
    df = simple_df.copy()
    mask = (df["country"] == "Germany") & (df["year"].isin([2005, 2006]))
    df.loc[mask, "electricity_demand"] = float("nan")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# partition_series
# ──────────────────────────────────────────────────────────────────────────────

class TestPartitionSeries:
    def test_correct_lengths(self, simple_df):
        ctx_y, ctx_v, ev_y, ev_v = partition_series(
            simple_df, "Germany", "electricity_demand", y_current=2015, L=10, H=3
        )
        assert len(ctx_y) == 10
        assert len(ctx_v) == 10
        assert len(ev_y) == 3
        assert len(ev_v) == 3

    def test_context_years_correct(self, simple_df):
        ctx_y, *_ = partition_series(
            simple_df, "Germany", "electricity_demand", y_current=2015, L=5, H=2
        )
        assert ctx_y == [2010, 2011, 2012, 2013, 2014]

    def test_eval_years_correct(self, simple_df):
        _, _, ev_y, _ = partition_series(
            simple_df, "Germany", "electricity_demand", y_current=2015, L=5, H=3
        )
        assert ev_y == [2015, 2016, 2017]

    def test_no_leakage(self, simple_df):
        ctx_y, ctx_v, ev_y, ev_v = partition_series(
            simple_df, "Germany", "electricity_demand", y_current=2015, L=5, H=3
        )
        assert max(ctx_y) < min(ev_y), "Context window must not overlap evaluation window."

    def test_gap_filling(self, df_with_nans):
        ctx_y, ctx_v, _, _ = partition_series(
            df_with_nans, "Germany", "electricity_demand", y_current=2015, L=10, H=3
        )
        assert not any(np.isnan(v) for v in ctx_v), "NaNs should be filled before returning."

    def test_insufficient_data_raises(self, simple_df):
        with pytest.raises(ValueError, match="Insufficient"):
            partition_series(
                simple_df, "Germany", "electricity_demand", y_current=2001, L=10, H=3
            )

    def test_unknown_country_raises(self, simple_df):
        with pytest.raises(ValueError, match="not found"):
            partition_series(
                simple_df, "Atlantis", "electricity_demand", y_current=2015, L=5, H=2
            )


# ──────────────────────────────────────────────────────────────────────────────
# _fill_gaps
# ──────────────────────────────────────────────────────────────────────────────

class TestFillGaps:
    def test_no_nans(self):
        sub = pd.DataFrame({"year": [2010, 2011, 2012], "val": [1.0, 2.0, 3.0]})
        result = _fill_gaps(sub, "val", [2010, 2011, 2012])
        assert result == [1.0, 2.0, 3.0]

    def test_internal_nan_interpolated(self):
        sub = pd.DataFrame({"year": [2010, 2011, 2012], "val": [1.0, float("nan"), 3.0]})
        result = _fill_gaps(sub, "val", [2010, 2011, 2012])
        assert result[1] == pytest.approx(2.0, abs=0.01)

    def test_missing_year_filled(self):
        sub = pd.DataFrame({"year": [2010, 2012], "val": [10.0, 30.0]})
        result = _fill_gaps(sub, "val", [2010, 2011, 2012])
        assert result[1] == pytest.approx(20.0, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# build_historical_database
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildHistoricalDatabase:
    def test_returns_list(self, simple_df):
        cands = build_historical_database(
            simple_df, "Germany", "electricity_demand", y_current=2018, L=5, H=3
        )
        assert isinstance(cands, list)
        assert len(cands) > 0

    def test_no_leakage_for_target_country(self, simple_df):
        cands = build_historical_database(
            simple_df, "Germany", "electricity_demand", y_current=2018, L=5, H=3
        )
        for c in cands:
            if c["country"] == "Germany":
                assert c["lookahead_years"][-1] < 2018, (
                    f"Germany lookahead bleeds into y_current=2018: {c['lookahead_years']}"
                )

    def test_window_lengths_consistent(self, simple_df):
        L, H = 5, 3
        cands = build_historical_database(
            simple_df, "Germany", "electricity_demand", y_current=2018, L=L, H=H
        )
        for c in cands:
            assert len(c["context_values"]) == L
            assert len(c["lookahead_values"]) == H


# ──────────────────────────────────────────────────────────────────────────────
# format_series
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatSeries:
    def test_basic_output(self):
        result = format_series([2010, 2011], [350.4, 362.1], unit="TWh")
        assert "2010:350.4" in result
        assert "2011:362.1" in result
        assert "TWh" in result

    def test_no_unit(self):
        result = format_series([2010], [42.0])
        assert "2010:42.0" in result
        assert "TWh" not in result


# ──────────────────────────────────────────────────────────────────────────────
# format_retrieved_segments
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatRetrievedSegments:
    def test_empty_returns_string(self):
        result = format_retrieved_segments([])
        assert "No similar" in result

    def test_single_segment(self):
        seg = {
            "country": "France",
            "context_years": [2000, 2001],
            "context_values": [100.0, 105.0],
            "lookahead_years": [2002, 2003],
            "lookahead_values": [110.0, 115.0],
            "correlation": 0.99,
        }
        result = format_retrieved_segments([seg])
        assert "France" in result
        assert "Analogue 1" in result
        assert "0.990" in result


# ──────────────────────────────────────────────────────────────────────────────
# format_history_log
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatHistoryLog:
    def test_renders_iterations(self):
        history = [
            {"iteration": 1, "instructions": "Extrapolate.", "predictions": [500.0], "mae": 12.3},
            {"iteration": 2, "instructions": "Adjust up.", "predictions": [510.0], "mae": 8.1},
        ]
        result = format_history_log(history)
        assert "Iteration 1" in result
        assert "Iteration 2" in result
        assert "12.3" in result
        assert "8.1" in result
