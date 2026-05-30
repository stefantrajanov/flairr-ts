"""
Unit tests – src/agents/retrieval.py
Runs without network access or API keys.
"""
import numpy as np
import pytest

from src.agents.retrieval import retrieve_similar_segments


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_candidate(
    country: str,
    start_year: int,
    ctx_values: list[float],
    lah_values: list[float],
) -> dict:
    L = len(ctx_values)
    H = len(lah_values)
    return {
        "country": country,
        "start_year": start_year,
        "end_year": start_year + L - 1,
        "context_years": list(range(start_year, start_year + L)),
        "context_values": ctx_values,
        "lookahead_years": list(range(start_year + L, start_year + L + H)),
        "lookahead_values": lah_values,
    }


@pytest.fixture()
def L():
    return 5


@pytest.fixture()
def candidates(L):
    """Three candidates: perfect positive correlation, negative, zero-variance."""
    x_target = [10.0, 12.0, 14.0, 16.0, 18.0]  # linear upward

    return [
        _make_candidate(
            "France", 2000,
            ctx_values=[20.0, 24.0, 28.0, 32.0, 36.0],  # r ≈ +1.0 (scaled same trend)
            lah_values=[40.0, 44.0, 48.0],
        ),
        _make_candidate(
            "Japan", 2000,
            ctx_values=[36.0, 32.0, 28.0, 24.0, 20.0],  # r ≈ -1.0 (inverse trend)
            lah_values=[16.0, 12.0, 8.0],
        ),
        _make_candidate(
            "Brazil", 2000,
            ctx_values=[50.0, 50.0, 50.0, 50.0, 50.0],  # zero variance → skipped
            lah_values=[50.0, 50.0, 50.0],
        ),
        _make_candidate(
            "India", 2000,
            ctx_values=[5.0, 5.2, 5.1, 5.3, 5.4],  # weak positive correlation
            lah_values=[5.5, 5.6, 5.7],
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRetrieveSimilarSegments:
    def test_returns_top_m(self, candidates, L):
        x_ctx = [10.0, 12.0, 14.0, 16.0, 18.0]
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=2)
        assert len(results) == 2

    def test_skips_zero_variance_candidate(self, candidates, L):
        x_ctx = [10.0, 12.0, 14.0, 16.0, 18.0]
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=4)
        countries = [r.location for r in results]
        assert "Brazil" not in countries, "Zero-variance candidates must be skipped."

    def test_highest_abs_correlation_first(self, candidates, L):
        x_ctx = [10.0, 12.0, 14.0, 16.0, 18.0]
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=3)
        # France (r≈+1) and Japan (r≈-1) both have |r|≈1 → ahead of India
        top_countries = {r.location for r in results[:2]}
        assert "France" in top_countries or "Japan" in top_countries

    def test_correlation_values_in_range(self, candidates, L):
        x_ctx = [10.0, 12.0, 14.0, 16.0, 18.0]
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=3)
        for r in results:
            assert -1.0 <= r.correlation <= 1.0

    def test_constant_context_returns_empty(self, candidates, L):
        x_ctx = [5.0] * L  # zero variance → should bail out
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=2)
        assert results == []

    def test_wrong_length_context_returns_empty(self, candidates, L):
        x_ctx = [1.0, 2.0]  # wrong length
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=2)
        assert results == []

    def test_m_larger_than_valid_candidates(self, candidates, L):
        x_ctx = [10.0, 12.0, 14.0, 16.0, 18.0]
        # Request more than exist (excluding zero-variance) → return what's available
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=99)
        assert len(results) <= len(candidates)
        assert all(r.location != "Brazil" for r in results)

    def test_segment_fields_populated(self, candidates, L):
        x_ctx = [10.0, 12.0, 14.0, 16.0, 18.0]
        results = retrieve_similar_segments(candidates, x_ctx, L=L, M=1)
        seg = results[0]
        assert seg.location
        assert len(seg.context_values) == L
        assert len(seg.lookahead_values) > 0
        assert seg.start_year > 0
