"""
tests/test_composite_objective.py — Unit tests for composite_objective function.

Tests the multi-dimensional scoring function directly with synthetic data.
Fast, no backtesting required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)

from strategies.optimizer_core import (
    composite_objective,
    _score_sharpe,
    _score_risk,
    _score_quality,
    _score_stability,
    _compute_monthly_win_rate,
    _compute_profit_concentration,
    MIN_TRADES_BY_FREQ,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_result(
    sharpe=1.0,
    max_drawdown=-0.10,
    n_trades=50,
    total_return=0.20,
    profit_concentration=0.4,
    monthly_win_rate=0.55,
):
    """Create a synthetic backtest result dict."""
    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "total_return": total_return,
        "profit_concentration": profit_concentration,
        "monthly_win_rate": monthly_win_rate,
    }


def _make_good_result():
    """A clearly good result: positive Sharpe, low drawdown, decent quality."""
    return _make_result(
        sharpe=1.5, max_drawdown=-0.08, n_trades=80,
        profit_concentration=0.35, monthly_win_rate=0.60,
    )


def _make_bad_result():
    """A clearly bad result: negative everything."""
    return _make_result(
        sharpe=-0.5, max_drawdown=-0.35, n_trades=80,
        profit_concentration=0.85, monthly_win_rate=0.30,
    )


# =====================================================================
# Test: Valid Results (positive Sharpe, reasonable drawdown)
# =====================================================================

class TestValidResults:
    """Test composite_objective with valid, positive results."""

    def test_single_good_result_returns_positive(self):
        result = _make_good_result()
        score = composite_objective([result], min_valid=1, freq="daily")
        assert score > 0, f"Good result should give positive score, got {score}"

    def test_single_good_result_in_range(self):
        result = _make_good_result()
        score = composite_objective([result], min_valid=1, freq="daily")
        assert 0 < score <= 10, f"Score should be in (0, 10], got {score}"

    def test_multiple_good_results_higher_than_threshold(self):
        results = [_make_good_result() for _ in range(5)]
        score = composite_objective(results, min_valid=3, freq="daily")
        assert score > 3.0, f"Multiple good results should score well, got {score}"

    def test_higher_sharpe_gives_higher_score(self):
        low = [_make_result(sharpe=0.5)]
        high = [_make_result(sharpe=2.0)]
        score_low = composite_objective(low, min_valid=1, freq="daily")
        score_high = composite_objective(high, min_valid=1, freq="daily")
        assert score_high > score_low, (
            f"Higher Sharpe should give higher score: "
            f"Sharpe=2.0 -> {score_high}, Sharpe=0.5 -> {score_low}"
        )

    def test_lower_drawdown_gives_higher_score(self):
        """Same Sharpe, different drawdown — lower drawdown should score higher."""
        low_dd = [_make_result(sharpe=1.0, max_drawdown=-0.05)]
        high_dd = [_make_result(sharpe=1.0, max_drawdown=-0.30)]
        score_low_dd = composite_objective(low_dd, min_valid=1, freq="daily")
        score_high_dd = composite_objective(high_dd, min_valid=1, freq="daily")
        assert score_low_dd > score_high_dd, (
            f"Lower drawdown should score higher: "
            f"dd=-5% -> {score_low_dd}, dd=-30% -> {score_high_dd}"
        )


# =====================================================================
# Test: All-Negative Results
# =====================================================================

class TestNegativeResults:
    """Test with all-negative Sharpe ratios."""

    def test_all_negative_sharpe_low_score(self):
        results = [_make_result(sharpe=-0.5) for _ in range(3)]
        score = composite_objective(results, min_valid=1, freq="daily")
        # Sharpe component is 0 for negative Sharpe, but risk/quality/stability
        # can still contribute. Score should be low.
        assert score < 5.0, f"All negative Sharpe should give low score, got {score}"

    def test_deeply_negative_sharpe(self):
        results = [_make_result(sharpe=-2.0, max_drawdown=-0.40)]
        score = composite_objective(results, min_valid=1, freq="daily")
        assert score < 3.0, f"Deeply negative Sharpe should give very low score, got {score}"

    def test_mixed_positive_negative(self):
        """Mix of positive and negative: should be lower than all-positive."""
        all_pos = [_make_result(sharpe=1.0) for _ in range(3)]
        mixed = [
            _make_result(sharpe=1.5),
            _make_result(sharpe=1.0),
            _make_result(sharpe=-0.3),
        ]
        score_pos = composite_objective(all_pos, min_valid=1, freq="daily")
        score_mixed = composite_objective(mixed, min_valid=1, freq="daily")
        assert score_pos > score_mixed, (
            f"All positive should score higher than mixed: "
            f"all_pos={score_pos}, mixed={score_mixed}"
        )


# =====================================================================
# Test: Insufficient Valid Results
# =====================================================================

class TestInsufficientResults:
    """Test min_valid threshold."""

    def test_below_min_valid_returns_negative(self):
        results = [_make_good_result(), _make_good_result()]
        score = composite_objective(results, min_valid=5, freq="daily")
        assert score == -10.0, f"Below min_valid should return -10.0, got {score}"

    def test_empty_results(self):
        score = composite_objective([], min_valid=1, freq="daily")
        assert score == -10.0, f"Empty results should return -10.0, got {score}"

    def test_all_failed_results(self):
        """Results with sharpe <= -900 are filtered out."""
        results = [_make_result(sharpe=-999.0) for _ in range(5)]
        score = composite_objective(results, min_valid=1, freq="daily")
        assert score == -10.0, f"All failed results should return -10.0, got {score}"

    def test_exactly_min_valid(self):
        """Exactly min_valid results should be accepted."""
        results = [_make_good_result() for _ in range(3)]
        score = composite_objective(results, min_valid=3, freq="daily")
        assert score > 0, f"Exactly min_valid results should return valid score, got {score}"

    def test_one_below_min_valid(self):
        """One less than min_valid should fail."""
        results = [_make_good_result() for _ in range(2)]
        score = composite_objective(results, min_valid=3, freq="daily")
        assert score == -10.0


# =====================================================================
# Test: tanh vs linear scoring mode
# =====================================================================

class TestScoringModes:
    """Verify tanh gives diminishing returns vs linear."""

    def test_tanh_diminishing_returns(self):
        """tanh mode: equal Sharpe increments yield decreasing score gains.
        Compare marginal gain from Sharpe 2->3 vs 0->1 (both 1.0 increments)."""
        score_0 = _score_sharpe(0.01, mode="tanh")  # near-zero but positive
        score_1 = _score_sharpe(1.0, mode="tanh")
        score_2 = _score_sharpe(2.0, mode="tanh")
        score_3 = _score_sharpe(3.0, mode="tanh")

        marginal_low = score_1 - score_0    # 0 -> 1
        marginal_high = score_3 - score_2   # 2 -> 3

        assert marginal_low > marginal_high, (
            f"tanh should have diminishing returns: "
            f"marginal(0->1)={marginal_low:.3f} should be > "
            f"marginal(2->3)={marginal_high:.3f}"
        )

    def test_linear_constant_marginal(self):
        """linear mode: equal Sharpe increments give equal score increments.
        Use equal-width intervals: 0.5->1.0 and 1.0->1.5 (both 0.5 wide)."""
        s05 = _score_sharpe(0.5, mode="linear")
        s10 = _score_sharpe(1.0, mode="linear")
        s15 = _score_sharpe(1.5, mode="linear")

        marginal_low = s10 - s05    # 0.5 -> 1.0
        marginal_high = s15 - s10   # 1.0 -> 1.5

        assert abs(marginal_high - marginal_low) < 0.01, (
            f"linear should have constant marginal for equal increments: "
            f"marginal(0.5->1.0)={marginal_low:.3f}, "
            f"marginal(1.0->1.5)={marginal_high:.3f}"
        )

    def test_tanh_vs_linear_at_high_sharpe(self):
        """At high Sharpe, tanh should give a LOWER score than linear.
        Actually tanh saturates at ~10, linear caps at 10. For Sharpe > ~1.5,
        tanh > linear because tanh(0.7*1.5)*10 ≈ 7.8 > 1.5*10/3 = 5.0.
        At Sharpe=3, tanh≈9.7, linear=10.0 → linear is higher at extremes."""
        score_tanh = _score_sharpe(3.0, mode="tanh")
        score_linear = _score_sharpe(3.0, mode="linear")
        # At Sharpe=3: tanh ≈ 9.7, linear = 10.0
        assert score_linear >= score_tanh, (
            f"At Sharpe=3: linear={score_linear:.2f} should be >= tanh={score_tanh:.2f}"
        )

    def test_tanh_vs_linear_at_moderate_sharpe(self):
        """At moderate Sharpe (~1.0), tanh gives HIGHER score than linear."""
        score_tanh = _score_sharpe(1.0, mode="tanh")
        score_linear = _score_sharpe(1.0, mode="linear")
        # tanh(0.7) * 10 ≈ 6.0, linear = 10/3 ≈ 3.3
        assert score_tanh > score_linear, (
            f"At Sharpe=1: tanh={score_tanh:.2f} should be > linear={score_linear:.2f}"
        )

    def test_scoring_mode_affects_composite(self):
        """composite_objective should give different scores for tanh vs linear."""
        results = [_make_result(sharpe=1.5)]
        score_tanh = composite_objective(results, min_valid=1, scoring_mode="tanh")
        score_linear = composite_objective(results, min_valid=1, scoring_mode="linear")
        assert score_tanh != score_linear, (
            f"Different scoring modes should give different results: "
            f"tanh={score_tanh}, linear={score_linear}"
        )

    def test_negative_sharpe_same_in_both_modes(self):
        """Both modes return 0 for negative Sharpe."""
        assert _score_sharpe(-1.0, mode="tanh") == 0.0
        assert _score_sharpe(-1.0, mode="linear") == 0.0
        assert _score_sharpe(0.0, mode="tanh") == 0.0
        assert _score_sharpe(0.0, mode="linear") == 0.0


# =====================================================================
# Test: Trade Count Filter
# =====================================================================

class TestTradeCountFilter:
    """Test that results with too few trades are filtered out."""

    def test_daily_min_trades(self):
        """Daily strategies need >= 10 trades."""
        assert MIN_TRADES_BY_FREQ["daily"] == 10

        # Below minimum: filtered out
        results = [_make_result(sharpe=2.0, n_trades=5)]
        score = composite_objective(results, min_valid=1, freq="daily")
        assert score == -10.0, (
            f"5 trades with daily freq should be filtered, got {score}"
        )

    def test_daily_above_minimum(self):
        """Trades above minimum should not be filtered."""
        results = [_make_result(sharpe=2.0, n_trades=20)]
        score = composite_objective(results, min_valid=1, freq="daily")
        assert score > 0, f"20 trades with daily freq should be valid, got {score}"

    def test_intraday_higher_minimum(self):
        """Intraday strategies need more trades."""
        assert MIN_TRADES_BY_FREQ["30min"] == 50

        # Enough for daily but not for 30min
        results = [_make_result(sharpe=2.0, n_trades=30)]
        score_daily = composite_objective(results, min_valid=1, freq="daily")
        score_30min = composite_objective(results, min_valid=1, freq="30min")

        assert score_daily > 0, "30 trades should pass for daily"
        assert score_30min == -10.0, "30 trades should fail for 30min"

    def test_none_freq_no_filter(self):
        """When freq is None, no trade count filtering."""
        results = [_make_result(sharpe=1.0, n_trades=1)]
        score = composite_objective(results, min_valid=1, freq=None)
        assert score > 0, f"No freq → no trade filter, got {score}"

    def test_none_n_trades_not_filtered(self):
        """When n_trades is None, result is not filtered out."""
        results = [_make_result(sharpe=1.0, n_trades=None)]
        score = composite_objective(results, min_valid=1, freq="daily")
        assert score > 0, f"n_trades=None should not be filtered, got {score}"

    def test_partial_filter(self):
        """Some results filtered, enough remain."""
        results = [
            _make_result(sharpe=1.5, n_trades=50),  # passes
            _make_result(sharpe=1.0, n_trades=50),  # passes
            _make_result(sharpe=2.0, n_trades=3),   # filtered (< 10)
        ]
        score = composite_objective(results, min_valid=2, freq="daily")
        assert score > 0, "Two passing results should be enough for min_valid=2"


# =====================================================================
# Test: Multi-Symbol Consistency Penalty
# =====================================================================

class TestMultiSymbolConsistency:
    """Test that having a negative-Sharpe symbol penalizes the score."""

    def test_all_positive_no_penalty(self):
        """All positive Sharpes: no consistency penalty."""
        results = [
            _make_result(sharpe=1.5),
            _make_result(sharpe=1.2),
            _make_result(sharpe=1.0),
        ]
        score = composite_objective(results, min_valid=1, freq="daily")
        # Compare with mean Sharpe 1.233 single result
        single = [_make_result(sharpe=float(np.mean([1.5, 1.2, 1.0])))]
        score_single = composite_objective(single, min_valid=1, freq="daily")
        # Multi-symbol all-positive should be similar to single with mean Sharpe
        # (may differ slightly due to other dimension aggregation)
        assert abs(score - score_single) < 2.0, (
            f"All positive multi-symbol score={score:.2f} should be close to "
            f"single mean-Sharpe score={score_single:.2f}"
        )

    def test_one_negative_penalizes(self):
        """One negative Sharpe symbol should reduce the score."""
        all_positive = [
            _make_result(sharpe=1.5),
            _make_result(sharpe=1.2),
            _make_result(sharpe=1.0),
        ]
        one_negative = [
            _make_result(sharpe=1.5),
            _make_result(sharpe=1.2),
            _make_result(sharpe=-0.5),
        ]
        score_pos = composite_objective(all_positive, min_valid=1, freq="daily")
        score_neg = composite_objective(one_negative, min_valid=1, freq="daily")
        assert score_pos > score_neg, (
            f"One negative symbol should reduce score: "
            f"all_pos={score_pos:.2f}, one_neg={score_neg:.2f}"
        )

    def test_worse_negative_penalizes_more(self):
        """A more negative worst symbol should penalize more."""
        mild_neg = [
            _make_result(sharpe=1.5),
            _make_result(sharpe=1.2),
            _make_result(sharpe=-0.2),
        ]
        severe_neg = [
            _make_result(sharpe=1.5),
            _make_result(sharpe=1.2),
            _make_result(sharpe=-1.5),
        ]
        score_mild = composite_objective(mild_neg, min_valid=1, freq="daily")
        score_severe = composite_objective(severe_neg, min_valid=1, freq="daily")
        assert score_mild > score_severe, (
            f"More negative worst symbol should penalize more: "
            f"mild(-0.2)={score_mild:.2f}, severe(-1.5)={score_severe:.2f}"
        )


# =====================================================================
# Test: Graceful Degradation (no equity curve data → neutral scores)
# =====================================================================

class TestGracefulDegradation:
    """Test behavior when optional data (equity curve, etc.) is missing."""

    def test_no_concentration_gives_neutral(self):
        """Missing profit_concentration should give neutral quality score (5.0)."""
        result = _make_result(sharpe=1.0, profit_concentration=None)
        score = composite_objective([result], min_valid=1)
        # Compare with explicit concentration
        result_with = _make_result(sharpe=1.0, profit_concentration=0.4)
        score_with = composite_objective([result_with], min_valid=1)
        # Score without concentration uses neutral 5.0; with 0.4 concentration gives ~8.5
        # So score_with should be higher (since 0.4 is good)
        assert score_with > score or abs(score_with - score) < 1.5

    def test_no_monthly_win_rate_gives_neutral(self):
        """Missing monthly_win_rate should give neutral stability score (5.0)."""
        result = _make_result(sharpe=1.0, monthly_win_rate=None)
        score = composite_objective([result], min_valid=1)
        assert score > 0, f"Missing win rate shouldn't break scoring, got {score}"

    def test_no_drawdown_gives_neutral(self):
        """Missing drawdown should give neutral risk score."""
        result = _make_result(sharpe=1.0, max_drawdown=None)
        score = composite_objective([result], min_valid=1)
        assert score > 0, f"Missing drawdown shouldn't break scoring, got {score}"

    def test_all_optional_missing(self):
        """Only Sharpe provided, everything else None."""
        result = {
            "sharpe": 1.5,
            "max_drawdown": None,
            "n_trades": None,
            "total_return": None,
            "profit_concentration": None,
            "monthly_win_rate": None,
        }
        score = composite_objective([result], min_valid=1)
        assert score > 0, f"Sharpe-only result should still score, got {score}"

    def test_minimal_result_dict(self):
        """Absolute minimum: just 'sharpe' key."""
        result = {"sharpe": 1.0}
        score = composite_objective([result], min_valid=1)
        assert score > 0, f"Minimal result should still work, got {score}"


# =====================================================================
# Test: Individual Scoring Functions
# =====================================================================

class TestScoringFunctions:
    """Test the individual dimension scoring functions."""

    # _score_sharpe
    def test_score_sharpe_zero(self):
        assert _score_sharpe(0.0) == 0.0

    def test_score_sharpe_negative(self):
        assert _score_sharpe(-1.0) == 0.0

    def test_score_sharpe_positive_tanh(self):
        score = _score_sharpe(1.0, mode="tanh")
        assert 5.0 < score < 7.0, f"Sharpe=1.0 tanh should be ~6.0, got {score}"

    def test_score_sharpe_high_tanh(self):
        score = _score_sharpe(3.0, mode="tanh")
        assert score > 9.0, f"Sharpe=3.0 tanh should be >9.0, got {score}"

    def test_score_sharpe_linear_cap(self):
        score = _score_sharpe(5.0, mode="linear")
        assert score == 10.0, f"Sharpe=5.0 linear should cap at 10.0, got {score}"

    # _score_risk
    def test_score_risk_low_dd(self):
        score = _score_risk(-0.03)
        assert score == 10.0, f"3% drawdown should score 10.0, got {score}"

    def test_score_risk_high_dd(self):
        score = _score_risk(-0.40)
        assert score == 0.0, f"40% drawdown should score 0.0, got {score}"

    def test_score_risk_moderate(self):
        score = _score_risk(-0.20)
        # Formula: 10 - (0.15/0.35)*10 ≈ 5.71
        assert 5.0 < score < 6.5, f"20% drawdown should score ~5.7, got {score}"

    # _score_quality
    def test_score_quality_low_concentration(self):
        score = _score_quality(0.2)
        assert score == 10.0, f"Low concentration should score 10.0, got {score}"

    def test_score_quality_high_concentration(self):
        score = _score_quality(0.95)
        assert score == 0.0, f"High concentration should score 0.0, got {score}"

    def test_score_quality_none(self):
        score = _score_quality(None)
        assert score == 5.0, f"None concentration should give neutral 5.0, got {score}"

    # _score_stability
    def test_score_stability_high_wr(self):
        score = _score_stability(0.70)
        assert score == 10.0, f"70% win rate should score 10.0, got {score}"

    def test_score_stability_low_wr(self):
        score = _score_stability(0.25)
        assert score == 0.0, f"25% win rate should score 0.0, got {score}"

    def test_score_stability_none(self):
        score = _score_stability(None)
        assert score == 5.0, f"None win rate should give neutral 5.0, got {score}"


# =====================================================================
# Test: Equity Curve Helpers
# =====================================================================

class TestEquityCurveHelpers:
    """Test _compute_monthly_win_rate and _compute_profit_concentration."""

    def test_monthly_wr_uptrend(self):
        """Steadily rising equity → high win rate."""
        equity = np.linspace(1_000_000, 1_500_000, 252)  # ~1 year daily
        wr = _compute_monthly_win_rate(equity)
        assert wr is not None
        assert wr > 0.8, f"Uptrend should have high win rate, got {wr}"

    def test_monthly_wr_downtrend(self):
        """Steadily falling equity → low win rate."""
        equity = np.linspace(1_000_000, 500_000, 252)
        wr = _compute_monthly_win_rate(equity)
        assert wr is not None
        assert wr < 0.2, f"Downtrend should have low win rate, got {wr}"

    def test_monthly_wr_too_short(self):
        """Very short equity curve → None."""
        equity = np.linspace(1_000_000, 1_100_000, 20)
        wr = _compute_monthly_win_rate(equity)
        assert wr is None

    def test_profit_concentration_even(self):
        """Steady linear growth → low concentration."""
        equity = np.linspace(1_000_000, 1_200_000, 252)
        conc = _compute_profit_concentration(equity)
        assert conc is not None
        assert conc < 0.5, f"Even growth should have low concentration, got {conc}"

    def test_profit_concentration_spike(self):
        """Flat except for one big spike → high concentration."""
        equity = np.ones(252) * 1_000_000
        # One big jump
        equity[125:] = 1_500_000
        conc = _compute_profit_concentration(equity)
        assert conc is not None
        assert conc > 0.8, f"Single spike should have high concentration, got {conc}"

    def test_profit_concentration_too_short(self):
        """Very short equity → None."""
        equity = np.array([1_000_000, 1_010_000])
        conc = _compute_profit_concentration(equity)
        assert conc is None
