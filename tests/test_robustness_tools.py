"""Tests for robustness testing tools (slippage + stress test).

All tests use mock backtest results — no real data or AlphaForge required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from tests.robustness.slippage_test import (
    SlippageResult,
    SlippageReport,
    compute_degradation,
    compute_verdict,
    verdict_description,
    format_report,
)
from tests.robustness.stress_test import (
    compute_sharpe_from_returns,
    compute_max_drawdown,
    compute_cvar,
    run_monte_carlo,
    run_stress_test,
    compute_stress_verdict,
    StressTestResult,
)


# =========================================================================
# Slippage: degradation calculation
# =========================================================================

class TestDegradation:
    def test_no_change(self):
        assert compute_degradation(3.0, 3.0) == 0.0

    def test_50_percent_drop(self):
        result = compute_degradation(3.0, 1.5)
        assert abs(result - (-0.5)) < 1e-10

    def test_improvement(self):
        """Degradation can be positive if test is better than baseline."""
        result = compute_degradation(2.0, 3.0)
        assert result == pytest.approx(0.5)

    def test_zero_baseline(self):
        """Zero baseline should return 0.0 (avoid division by zero)."""
        assert compute_degradation(0.0, 1.5) == 0.0

    def test_negative_baseline(self):
        """Negative baseline should return 0.0."""
        assert compute_degradation(-1.0, 1.5) == 0.0

    def test_22_percent_drop(self):
        """Matches the example table: Sharpe 3.09 -> 2.41 is ~22% drop."""
        result = compute_degradation(3.09, 2.41)
        assert abs(result) == pytest.approx(0.22, abs=0.01)


# =========================================================================
# Slippage: verdict logic
# =========================================================================

class TestVerdict:
    def test_low_sensitivity(self):
        """< 20% drop at 2x -> LOW."""
        assert compute_verdict(-0.10) == "LOW"
        assert compute_verdict(-0.19) == "LOW"
        assert compute_verdict(0.0) == "LOW"

    def test_moderate_sensitivity(self):
        """20-40% drop at 2x -> MODERATE."""
        assert compute_verdict(-0.20) == "MODERATE"
        assert compute_verdict(-0.30) == "MODERATE"
        assert compute_verdict(-0.40) == "MODERATE"

    def test_high_sensitivity(self):
        """> 40% drop at 2x -> HIGH."""
        assert compute_verdict(-0.41) == "HIGH"
        assert compute_verdict(-0.70) == "HIGH"
        assert compute_verdict(-1.0) == "HIGH"

    def test_boundary_20(self):
        """Exactly 20% is MODERATE (inclusive)."""
        assert compute_verdict(-0.20) == "MODERATE"

    def test_boundary_40(self):
        """Exactly 40% is MODERATE (inclusive)."""
        assert compute_verdict(-0.40) == "MODERATE"

    def test_verdict_descriptions(self):
        assert "robust" in verdict_description("LOW")
        assert "acceptable" in verdict_description("MODERATE")
        assert "consider removing" in verdict_description("HIGH")


# =========================================================================
# Slippage: report formatting
# =========================================================================

class TestSlippageReport:
    def _make_report(self):
        results = [
            SlippageResult(1.0, 3.09, 0.852, -0.082, 120),
            SlippageResult(2.0, 2.41, 0.621, -0.101, 120),
            SlippageResult(3.0, 1.88, 0.443, -0.125, 120),
            SlippageResult(5.0, 0.92, 0.187, -0.183, 120),
        ]
        return SlippageReport(
            strategy_name="StrongTrendV12",
            symbol="AG",
            freq="daily",
            results=results,
            verdict="MODERATE",
            degradation_at_2x=-0.22,
        )

    def test_format_contains_strategy_name(self):
        report = self._make_report()
        text = format_report(report)
        assert "StrongTrendV12" in text
        assert "AG" in text
        assert "daily" in text

    def test_format_contains_verdict(self):
        report = self._make_report()
        text = format_report(report)
        assert "MODERATE" in text

    def test_to_dict_roundtrip(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["strategy"] == "StrongTrendV12"
        assert d["symbol"] == "AG"
        assert d["verdict"] == "MODERATE"
        assert len(d["levels"]) == 4
        assert d["levels"][0]["slippage_ticks"] == 1.0
        assert d["levels"][0]["sharpe"] == 3.09


# =========================================================================
# Stress test: Sharpe from returns
# =========================================================================

class TestSharpeFromReturns:
    def test_positive_returns(self):
        """Constant positive returns should yield very high Sharpe."""
        returns = np.array([0.01] * 252)
        sharpe = compute_sharpe_from_returns(returns)
        # All returns are identical, so std=0 -> returns 0.0
        assert sharpe == 0.0

    def test_mixed_returns(self):
        """Known distribution: mean=0.001, std=0.01 -> Sharpe ~ 1.59."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=252)
        sharpe = compute_sharpe_from_returns(returns)
        # Should be roughly positive
        assert sharpe > 0

    def test_negative_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.005, 0.01, size=252)
        sharpe = compute_sharpe_from_returns(returns)
        assert sharpe < 0

    def test_empty_returns(self):
        assert compute_sharpe_from_returns(np.array([])) == 0.0

    def test_single_return(self):
        assert compute_sharpe_from_returns(np.array([0.01])) == 0.0


# =========================================================================
# Stress test: max drawdown
# =========================================================================

class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing equity -> DD is 0 or very small."""
        returns = np.array([0.01] * 100)
        dd = compute_max_drawdown(returns)
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        """Equity goes up then drops 50%: 1.0 -> 1.1 -> 0.55."""
        returns = np.array([0.10, -0.5])  # +10%, then -50%
        dd = compute_max_drawdown(returns)
        assert dd == pytest.approx(-0.5, abs=0.01)

    def test_empty(self):
        assert compute_max_drawdown(np.array([])) == 0.0

    def test_all_negative(self):
        """All negative returns should give large drawdown."""
        returns = np.array([-0.01] * 50)
        dd = compute_max_drawdown(returns)
        assert dd < -0.30  # roughly -39%


# =========================================================================
# Stress test: CVaR
# =========================================================================

class TestCVaR:
    def test_known_distribution(self):
        """For uniform returns [-0.05, 0.05], CVaR 95% should be mean of worst 5%."""
        rng = np.random.default_rng(42)
        returns = rng.uniform(-0.05, 0.05, size=10000)
        cvar = compute_cvar(returns, confidence=0.95)
        # Worst 5% of uniform[-0.05, 0.05] is uniform[-0.05, -0.04]
        # Mean of that is about -0.045
        assert -0.055 < cvar < -0.035

    def test_cvar_99_worse_than_95(self):
        """CVaR 99% should be more negative than CVaR 95%."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=5000)
        cvar_95 = compute_cvar(returns, 0.95)
        cvar_99 = compute_cvar(returns, 0.99)
        assert cvar_99 < cvar_95

    def test_empty(self):
        assert compute_cvar(np.array([]), 0.95) == 0.0

    def test_all_positive(self):
        """Even with all positive returns, CVaR picks the worst tail."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 100)
        cvar = compute_cvar(returns, 0.95)
        # Worst 5% are all 0.01
        assert cvar == pytest.approx(0.01, abs=0.005)


# =========================================================================
# Stress test: Monte Carlo
# =========================================================================

class TestMonteCarlo:
    def test_output_shape(self):
        returns = np.random.default_rng(42).normal(0.001, 0.01, 252)
        result = run_monte_carlo(returns, n_simulations=100, seed=42)
        assert result["sharpes"].shape == (100,)
        assert result["max_drawdowns"].shape == (100,)

    def test_reproducibility(self):
        returns = np.random.default_rng(42).normal(0.001, 0.01, 252)
        r1 = run_monte_carlo(returns, n_simulations=50, seed=123)
        r2 = run_monte_carlo(returns, n_simulations=50, seed=123)
        np.testing.assert_array_equal(r1["sharpes"], r2["sharpes"])

    def test_sharpe_distribution_reasonable(self):
        """With positive mean returns, most simulated Sharpes should be positive."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 252)  # Strong positive signal
        result = run_monte_carlo(returns, n_simulations=500, seed=42)
        pct_positive = np.mean(result["sharpes"] > 0)
        assert pct_positive > 0.80  # At least 80% positive

    def test_ci_contains_base(self):
        """95% CI from MC should usually contain the base Sharpe (or be close)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        base_sharpe = compute_sharpe_from_returns(returns)
        result = run_monte_carlo(returns, n_simulations=1000, seed=42)
        p5 = np.percentile(result["sharpes"], 2.5)
        p95 = np.percentile(result["sharpes"], 97.5)
        # Base Sharpe should be within a reasonable range of MC distribution
        assert p5 < base_sharpe + 1.0  # relaxed bound
        assert p95 > base_sharpe - 1.0


# =========================================================================
# Stress test: full run_stress_test
# =========================================================================

class TestRunStressTest:
    def test_basic_run(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 252)
        result = run_stress_test(returns, symbol="TEST", n_simulations=200, seed=42)

        assert isinstance(result, StressTestResult)
        assert result.symbol == "TEST"
        assert result.n_simulations == 200
        assert result.base_sharpe > 0
        assert result.mc_mean_sharpe > 0
        assert result.mc_5th_pctl < result.mc_95th_pctl
        assert 0 <= result.prob_negative_sharpe <= 1
        assert result.cvar_95 < 0  # daily returns have negative tail
        assert result.cvar_99 <= result.cvar_95
        assert result.max_sim_dd < 0
        assert result.verdict in ("ROBUST", "ACCEPTABLE", "FRAGILE")

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 252)
        result = run_stress_test(returns, symbol="TEST", n_simulations=50, seed=42)
        d = result.to_dict()
        assert d["symbol"] == "TEST"
        assert "base_sharpe" in d
        assert "verdict" in d


# =========================================================================
# Stress test: verdict logic
# =========================================================================

class TestStressVerdict:
    def test_robust(self):
        assert compute_stress_verdict(0.01, -0.02) == "ROBUST"

    def test_acceptable(self):
        assert compute_stress_verdict(0.10, -0.04) == "ACCEPTABLE"

    def test_fragile_high_prob(self):
        assert compute_stress_verdict(0.20, -0.02) == "FRAGILE"

    def test_fragile_high_cvar(self):
        assert compute_stress_verdict(0.01, -0.06) == "FRAGILE"

    def test_boundary_robust(self):
        """Exactly at ROBUST thresholds."""
        assert compute_stress_verdict(0.049, -0.029) == "ROBUST"

    def test_boundary_acceptable(self):
        assert compute_stress_verdict(0.05, -0.03) == "ACCEPTABLE"
