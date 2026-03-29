"""Tests for drawdown attribution module."""
import sys
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pytest

from attribution.drawdown import (
    DrawdownPeriod,
    DrawdownAttribution,
    find_max_drawdown,
    _compute_regime_during_period,
    _generate_report,
)


# =========================================================================
# find_max_drawdown
# =========================================================================

class TestFindMaxDrawdown:
    """Tests for the max drawdown detection function."""

    def test_simple_drawdown(self):
        """Basic V-shaped drawdown should be detected correctly."""
        equity = np.array([100, 110, 120, 100, 90, 95, 105])
        dd = find_max_drawdown(equity)

        assert dd.drawdown_pct < 0, "Drawdown should be negative"
        # Peak is 120, trough is 90 -> DD = (90-120)/120 = -25%
        assert abs(dd.drawdown_pct - (-25.0)) < 0.1
        assert dd.peak_equity == 120.0
        assert dd.trough_equity == 90.0

    def test_monotonically_increasing(self):
        """A monotonically increasing curve should have zero drawdown."""
        equity = np.array([100, 101, 102, 103, 104, 105])
        dd = find_max_drawdown(equity)

        assert dd.drawdown_pct == 0.0, "No drawdown in uptrend"
        assert dd.duration_days == 0

    def test_monotonically_decreasing(self):
        """A monotonically decreasing curve: DD from bar 0 to last bar."""
        equity = np.array([100, 90, 80, 70, 60])
        dd = find_max_drawdown(equity)

        # Peak=100, trough=60 -> DD = (60-100)/100 = -40%
        assert abs(dd.drawdown_pct - (-40.0)) < 0.1
        assert dd.peak_equity == 100.0
        assert dd.trough_equity == 60.0

    def test_single_bar(self):
        """A single-bar equity curve should return zero drawdown."""
        equity = np.array([1000.0])
        dd = find_max_drawdown(equity)
        assert dd.drawdown_pct == 0.0
        assert dd.peak_equity == 1000.0

    def test_two_bars_no_dd(self):
        """Two bars with increase -> no drawdown."""
        equity = np.array([100, 110])
        dd = find_max_drawdown(equity)
        assert dd.drawdown_pct == 0.0

    def test_two_bars_with_dd(self):
        """Two bars with decrease -> drawdown detected."""
        equity = np.array([100, 80])
        dd = find_max_drawdown(equity)
        assert abs(dd.drawdown_pct - (-20.0)) < 0.1

    def test_multiple_drawdowns_finds_worst(self):
        """When there are multiple drawdowns, should find the worst one."""
        # DD1: 100->90 = -10%, DD2: 120->85 = -29.2%
        equity = np.array([100, 90, 95, 120, 100, 85, 110])
        dd = find_max_drawdown(equity)

        # Worst is from 120 -> 85
        assert dd.peak_equity == 120.0
        assert dd.trough_equity == 85.0
        expected_dd = (85 - 120) / 120 * 100  # -29.17%
        assert abs(dd.drawdown_pct - expected_dd) < 0.1

    def test_with_dates(self):
        """Dates should be correctly mapped to drawdown period."""
        equity = np.array([100, 110, 120, 100, 90, 95, 105])
        dates = [
            "2025-01-01", "2025-01-02", "2025-01-03",
            "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
        ]
        dd = find_max_drawdown(equity, dates=dates)

        assert dd.start_date == "2025-01-03"  # peak at index 2
        assert dd.end_date == "2025-01-05"    # trough at index 4
        assert dd.duration_days == 2

    def test_flat_equity(self):
        """Flat equity curve should have zero drawdown."""
        equity = np.array([100, 100, 100, 100])
        dd = find_max_drawdown(equity)
        assert dd.drawdown_pct == 0.0


# =========================================================================
# Strategy contribution calculation
# =========================================================================

class TestStrategyContributions:
    """Tests for strategy contribution logic using mock equity curves."""

    def test_equal_contributors(self):
        """Two strategies with equal loss should each contribute ~50%."""
        # Strategy A: 100 -> 80 (loss = 20)
        # Strategy B: 100 -> 80 (loss = 20)
        # Portfolio (50/50): 100 -> 80 (loss = 20)
        ec_a = np.array([100, 95, 80])
        ec_b = np.array([100, 90, 80])
        weights = {"A": 0.5, "B": 0.5}

        # Peak at index 0, trough at index 2
        total_dd = -20.0
        contrib_a = (ec_a[2] - ec_a[0]) * weights["A"]
        contrib_b = (ec_b[2] - ec_b[0]) * weights["B"]

        pct_a = contrib_a / total_dd * 100
        pct_b = contrib_b / total_dd * 100

        assert abs(pct_a - 50.0) < 0.1
        assert abs(pct_b - 50.0) < 0.1

    def test_unequal_contributors(self):
        """Strategy with larger loss should have larger contribution %."""
        ec_a = np.array([100, 95, 70])   # loss = 30
        ec_b = np.array([100, 98, 90])   # loss = 10
        weights = {"A": 0.5, "B": 0.5}

        total_dd = (70 - 100) * 0.5 + (90 - 100) * 0.5  # -20
        contrib_a = (ec_a[2] - ec_a[0]) * weights["A"]   # -15
        contrib_b = (ec_b[2] - ec_b[0]) * weights["B"]   # -5

        pct_a = contrib_a / total_dd * 100
        pct_b = contrib_b / total_dd * 100

        assert pct_a > pct_b
        assert abs(pct_a - 75.0) < 0.1
        assert abs(pct_b - 25.0) < 0.1

    def test_one_strategy_profitable(self):
        """One strategy gains during drawdown — it should show negative contribution %."""
        ec_a = np.array([100, 95, 70])   # loss = 30
        ec_b = np.array([100, 102, 110]) # gain = 10
        weights = {"A": 0.5, "B": 0.5}

        total_dd = (70 - 100) * 0.5 + (110 - 100) * 0.5  # -10
        contrib_a = (ec_a[2] - ec_a[0]) * weights["A"]    # -15
        contrib_b = (ec_b[2] - ec_b[0]) * weights["B"]    # +5

        pct_a = contrib_a / total_dd * 100  # 150%
        pct_b = contrib_b / total_dd * 100  # -50%

        assert pct_a > 100  # Contributed more than 100% of DD
        assert pct_b < 0    # Offset some of the DD


# =========================================================================
# Regime tagging (requires indicator imports)
# =========================================================================

class TestRegimeDuringDrawdown:
    """Tests for regime tagging during drawdown period."""

    def test_regime_returns_expected_keys(self):
        """Regime dict should contain trend, volume, volatility keys."""
        n = 300
        np.random.seed(42)
        highs = np.cumsum(np.random.randn(n)) + 5000
        lows = highs - np.abs(np.random.randn(n)) * 10
        closes = (highs + lows) / 2
        volumes = np.abs(np.random.randn(n)) * 10000 + 5000

        regime = _compute_regime_during_period(highs, lows, closes, volumes, 100, 200)

        assert "trend" in regime
        assert "volume" in regime
        assert "volatility" in regime
        assert regime["trend"] in ("strong", "weak", "none", "unknown")
        assert regime["volume"] in ("active", "normal", "quiet", "unknown")
        assert regime["volatility"] in ("high", "normal", "low", "unknown")

    def test_regime_detail_strings(self):
        """Regime should include detail strings for each dimension."""
        n = 300
        np.random.seed(42)
        highs = np.cumsum(np.random.randn(n)) + 5000
        lows = highs - np.abs(np.random.randn(n)) * 10
        closes = (highs + lows) / 2
        volumes = np.abs(np.random.randn(n)) * 10000 + 5000

        regime = _compute_regime_during_period(highs, lows, closes, volumes, 100, 200)

        assert "trend_detail" in regime
        assert "volume_detail" in regime
        assert "volatility_detail" in regime


# =========================================================================
# Report generation
# =========================================================================

class TestReportGeneration:
    """Tests for Markdown report format."""

    def test_report_contains_required_sections(self, tmp_path):
        """Report should contain all required sections."""
        period = DrawdownPeriod(
            start_date="2025-06-15", end_date="2025-07-22",
            duration_days=37, drawdown_pct=-12.8,
            peak_equity=3384000, trough_equity=2951000,
        )
        attribution = DrawdownAttribution(
            period=period,
            strategy_contributions={
                "v12": {"pnl": -168000, "pct_of_dd": 41, "weight": 0.25},
                "v11": {"pnl": -126000, "pct_of_dd": 30, "weight": 0.20},
            },
            regime_during_dd={
                "trend": "weak", "trend_detail": "ADX avg = 18.3",
                "volume": "quiet", "volume_detail": "avg 0.6x of 20-day mean",
                "volatility": "low", "volatility_detail": "ATR percentile 22%",
            },
            conclusion="Test conclusion.",
        )

        output_path = str(tmp_path / "drawdown_AG.md")
        _generate_report(attribution, "AG", output_path)

        with open(output_path) as f:
            content = f.read()

        assert "# Portfolio Drawdown Attribution" in content
        assert "Maximum Drawdown" in content
        assert "Strategy Contributions" in content
        assert "Market Regime" in content
        assert "Conclusion" in content
        assert "v12" in content
        assert "v11" in content
        assert "-12.80%" in content
        assert "37 days" in content

    def test_report_table_format(self, tmp_path):
        """Report should have properly formatted Markdown table."""
        period = DrawdownPeriod(
            start_date="2025-01-01", end_date="2025-02-01",
            duration_days=31, drawdown_pct=-5.0,
            peak_equity=1000000, trough_equity=950000,
        )
        attribution = DrawdownAttribution(
            period=period,
            strategy_contributions={
                "v1": {"pnl": -50000, "pct_of_dd": 100, "weight": 1.0},
            },
            regime_during_dd={
                "trend": "strong", "trend_detail": "ADX avg = 30",
                "volume": "active", "volume_detail": "1.5x",
                "volatility": "high", "volatility_detail": "ATR 80%",
            },
            conclusion="Single strategy test.",
        )

        output_path = str(tmp_path / "drawdown_test.md")
        _generate_report(attribution, "TEST", output_path)

        with open(output_path) as f:
            content = f.read()

        # Check table header markers
        assert "|----------|" in content
        assert "| v1 |" in content
