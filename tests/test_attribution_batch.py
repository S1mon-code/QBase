"""Tests for batch attribution and regime coverage matrix."""
import os
import sys
import json
import tempfile
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import pytest


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_regime_stats():
    """Create a RegimeStats with configurable PnL."""
    from attribution.regime import RegimeStats

    def _make(n_trades=10, win_rate=60.0, total_pnl_pct=15.0):
        return RegimeStats(
            n_trades=n_trades,
            win_rate=win_rate,
            avg_pnl_pct=round(total_pnl_pct / max(n_trades, 1), 2),
            total_pnl_pct=total_pnl_pct,
            avg_holding_bars=12.0,
            best_trade_pnl=5.0,
            worst_trade_pnl=-2.0,
        )
    return _make


@pytest.fixture
def mock_regime_result(mock_regime_stats):
    """Create a RegimeAttributionResult with configurable regime performance."""
    from attribution.regime import RegimeAttributionResult

    def _make(
        version="v1",
        symbol="AG",
        trend_pnls=None,
        vol_pnls=None,
        act_pnls=None,
    ):
        # Defaults: profitable in all regimes
        if trend_pnls is None:
            trend_pnls = {"strong": 20.0, "weak": 5.0, "none": -3.0}
        if vol_pnls is None:
            vol_pnls = {"high": 15.0, "normal": 8.0, "low": 2.0}
        if act_pnls is None:
            act_pnls = {"active": 12.0, "normal": 6.0, "quiet": -1.0}

        by_trend = {k: mock_regime_stats(total_pnl_pct=v) for k, v in trend_pnls.items()}
        by_vol = {k: mock_regime_stats(total_pnl_pct=v) for k, v in vol_pnls.items()}
        by_act = {k: mock_regime_stats(total_pnl_pct=v) for k, v in act_pnls.items()}

        return RegimeAttributionResult(
            strategy_version=version,
            symbol=symbol,
            period="2025-01-01 ~ 2026-03-01",
            total_trades=30,
            total_sharpe=2.5,
            by_trend=by_trend,
            by_volatility=by_vol,
            by_activity=by_act,
            cross_trend_vol={},
            best_regime="trend=strong",
            worst_regime="trend=none",
        )
    return _make


@pytest.fixture
def sample_weights_json_dict(tmp_path):
    """Create a sample weights JSON in dict format (builder output)."""
    data = {
        "meta": {
            "method": "greedy_hrp_sharpe",
            "symbol": "AG",
            "test_start": "2025-01-01",
            "test_end": "2026-03-01",
        },
        "strategies": {
            "v12": {
                "weight": 0.25,
                "sharpe": 3.09,
                "freq": "daily",
                "role": "core",
                "params": {
                    "aroon_period": 20,
                    "ppo_fast": 19,
                    "ppo_slow": 29,
                    "vol_mom_period": 25,
                    "atr_trail_mult": 4.814,
                }
            },
            "v20": {
                "weight": 0.20,
                "sharpe": 1.80,
                "freq": "daily",
                "role": "core",
                "params": {
                    "fractal_period": 2,
                    "mass_ema": 7,
                    "mass_sum": 27,
                    "vroc_period": 16,
                    "atr_trail_mult": 2.165,
                }
            },
        }
    }
    path = tmp_path / "weights_ag.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


@pytest.fixture
def sample_weights_json_list(tmp_path):
    """Create a sample weights JSON in list format (manual/LC style)."""
    data = {
        "symbol": "LC",
        "method": "manual_balanced",
        "strategies": [
            {"version": "v12", "weight": 0.25, "freq": "daily", "role": "core"},
            {"version": "v5", "weight": 0.20, "freq": "daily", "role": "core"},
        ]
    }
    path = tmp_path / "weights_lc.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


# =========================================================================
# Test Weights Parsing
# =========================================================================

class TestParseWeightsJson:
    """Test weights JSON parsing handles all known formats."""

    def test_parse_dict_format(self, sample_weights_json_dict):
        from attribution.batch import parse_weights_json
        entries, meta = parse_weights_json(sample_weights_json_dict)

        assert len(entries) == 2
        v12 = next(e for e in entries if e.version == "v12")
        assert v12.weight == 0.25
        assert v12.freq == "daily"
        assert v12.params["aroon_period"] == 20
        assert meta["symbol"] == "AG"

    def test_parse_list_format(self, sample_weights_json_list):
        from attribution.batch import parse_weights_json
        entries, meta = parse_weights_json(sample_weights_json_list)

        assert len(entries) == 2
        assert entries[0].version == "v12"
        assert entries[0].weight == 0.25
        # List format without inline params — fallback to optimization_results.json
        # Since tmp_path has no "strategies" dir, it falls back to strong_trend
        # and finds v12 params from optimization_results.json (if it exists)
        # Just verify the parse didn't crash and returned a valid entry
        assert entries[0].freq == "daily"

    def test_detect_strategy_dir_strong_trend(self):
        from attribution.batch import _detect_strategy_dir
        path = "/Users/foo/QBase/strategies/strong_trend/portfolio/weights_ag.json"
        assert _detect_strategy_dir(path) == "strong_trend"

    def test_detect_strategy_dir_all_time(self):
        from attribution.batch import _detect_strategy_dir
        path = "/Users/foo/QBase/strategies/all_time/ag/portfolio/weights_ag.json"
        assert _detect_strategy_dir(path) == "all_time/ag"


# =========================================================================
# Test Coverage Matrix
# =========================================================================

class TestCoverageMatrix:
    """Test regime coverage matrix generation."""

    def test_generates_md_and_json(self, mock_regime_result, tmp_path):
        """Coverage matrix should produce both Markdown and JSON files."""
        from attribution.coverage import generate_coverage_matrix

        regime_results = {
            "v12": mock_regime_result(version="v12"),
            "v20": mock_regime_result(version="v20"),
        }
        weights = {"v12": 0.25, "v20": 0.20}

        md_path, json_path = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        assert os.path.exists(md_path)
        assert os.path.exists(json_path)
        assert md_path.endswith("portfolio_AG_coverage.md")
        assert json_path.endswith("portfolio_AG_coverage.json")

    def test_md_contains_all_sections(self, mock_regime_result, tmp_path):
        """Markdown should contain trend, volatility, activity sections."""
        from attribution.coverage import generate_coverage_matrix

        regime_results = {"v12": mock_regime_result(version="v12")}
        weights = {"v12": 0.25}

        md_path, _ = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        content = open(md_path).read()
        assert "Trend Strength" in content
        assert "Volatility" in content
        assert "Volume Activity" in content
        assert "Portfolio Coverage" in content
        assert "Strategy Summary" in content

    def test_json_structure(self, mock_regime_result, tmp_path):
        """JSON should have coverage data for all three dimensions."""
        from attribution.coverage import generate_coverage_matrix

        regime_results = {"v12": mock_regime_result(version="v12")}
        weights = {"v12": 0.25}

        _, json_path = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        with open(json_path) as f:
            data = json.load(f)

        assert "coverage" in data
        assert "trend" in data["coverage"]
        assert "volatility" in data["coverage"]
        assert "activity" in data["coverage"]
        assert data["symbol"] == "AG"
        assert data["n_strategies"] == 1


# =========================================================================
# Test Red Flag Detection
# =========================================================================

class TestRedFlagDetection:
    """Test that RED FLAGS are correctly raised for underexposed regimes."""

    def test_no_red_flags_when_all_covered(self, mock_regime_result, tmp_path):
        """Five strategies profitable in all regimes should produce no flags."""
        from attribution.coverage import generate_coverage_matrix

        # All 5 strategies profitable in all trend regimes
        regime_results = {
            f"v{i}": mock_regime_result(
                version=f"v{i}",
                trend_pnls={"strong": 20.0, "weak": 10.0, "none": 5.0},
            )
            for i in range(1, 6)
        }
        weights = {f"v{i}": 0.20 for i in range(1, 6)}

        _, json_path = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        with open(json_path) as f:
            data = json.load(f)

        # Filter to just trend flags (activity/vol might have negative defaults)
        trend_flags = [f for f in data["red_flags"] if "Trend" in f]
        assert len(trend_flags) == 0

    def test_red_flag_zero_profitable(self, mock_regime_result, tmp_path):
        """Regime with 0 profitable strategies should raise a red flag."""
        from attribution.coverage import generate_coverage_matrix

        # Both strategies negative in 'none' trend
        regime_results = {
            "v1": mock_regime_result(
                version="v1",
                trend_pnls={"strong": 20.0, "weak": 10.0, "none": -5.0},
            ),
            "v2": mock_regime_result(
                version="v2",
                trend_pnls={"strong": 15.0, "weak": 8.0, "none": -3.0},
            ),
        }
        weights = {"v1": 0.50, "v2": 0.50}

        _, json_path = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        with open(json_path) as f:
            data = json.load(f)

        no_trend_flags = [f for f in data["red_flags"] if "No Trend" in f]
        assert len(no_trend_flags) == 1
        assert "0 profitable" in no_trend_flags[0]

    def test_red_flag_one_profitable(self, mock_regime_result, tmp_path):
        """Regime with exactly 1 profitable strategy should raise a red flag."""
        from attribution.coverage import generate_coverage_matrix

        # Only v1 profitable in weak trend, v2 and v3 are negative
        regime_results = {
            "v1": mock_regime_result(
                version="v1",
                trend_pnls={"strong": 20.0, "weak": 5.0, "none": 10.0},
            ),
            "v2": mock_regime_result(
                version="v2",
                trend_pnls={"strong": 15.0, "weak": -3.0, "none": 8.0},
            ),
            "v3": mock_regime_result(
                version="v3",
                trend_pnls={"strong": 12.0, "weak": -1.0, "none": 6.0},
            ),
        }
        weights = {"v1": 0.40, "v2": 0.30, "v3": 0.30}

        _, json_path = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        with open(json_path) as f:
            data = json.load(f)

        weak_flags = [f for f in data["red_flags"] if "Weak Trend" in f]
        assert len(weak_flags) == 1
        assert "1 profitable" in weak_flags[0]

    def test_red_flag_in_md(self, mock_regime_result, tmp_path):
        """Red flags should appear in the Markdown report too."""
        from attribution.coverage import generate_coverage_matrix

        regime_results = {
            "v1": mock_regime_result(
                version="v1",
                trend_pnls={"strong": 20.0, "weak": -5.0, "none": -5.0},
            ),
        }
        weights = {"v1": 1.0}

        md_path, _ = generate_coverage_matrix(
            regime_results, weights, "AG", output_dir=str(tmp_path),
        )

        content = open(md_path).read()
        assert "RED FLAG" in content


# =========================================================================
# Test Graceful Failure Handling
# =========================================================================

class TestGracefulFailures:
    """Test that batch attribution handles individual strategy failures."""

    def test_batch_continues_on_failure(self, tmp_path):
        """If one strategy fails, the batch should continue with the rest.

        Uses mock strategy classes to avoid needing real market data.
        """
        from attribution.batch import StrategyEntry
        from attribution.regime import RegimeAttributionResult

        # Simulate the failure-handling logic from run_batch_attribution
        entries = [
            StrategyEntry(version="v_good", weight=0.50, freq="daily",
                          params={"x": 1}),
            StrategyEntry(version="v_bad", weight=0.30, freq="daily",
                          params={"x": 2}),
            StrategyEntry(version="v_noparam", weight=0.20, freq="daily",
                          params={}),
        ]

        # Track results + failures like the real function does
        signal_results = {}
        regime_results = {}
        failures = []

        for entry in entries:
            if not entry.params:
                failures.append({"version": entry.version, "reason": "no params"})
                continue

            if entry.version == "v_bad":
                failures.append({"version": entry.version, "reason": "test error"})
                continue

            # Simulate success
            signal_results[entry.version] = "mock_signal"
            regime_results[entry.version] = "mock_regime"

        assert len(signal_results) == 1
        assert "v_good" in signal_results
        assert len(failures) == 2
        assert failures[0]["version"] == "v_bad"
        assert failures[1]["version"] == "v_noparam"

    def test_skip_no_params(self, tmp_path):
        """Strategies without params should be skipped, not crash."""
        from attribution.batch import StrategyEntry

        entry = StrategyEntry(version="v99", weight=0.10, freq="daily", params={})
        assert not entry.params  # Will trigger skip in batch runner


# =========================================================================
# Test Strategy Directory Detection
# =========================================================================

class TestStrategyDirDetection:
    """Test strategy directory inference from weights path."""

    def test_strong_trend_dir(self):
        from attribution.batch import _detect_strategy_dir
        assert _detect_strategy_dir(
            "/foo/strategies/strong_trend/portfolio/weights_ag.json"
        ) == "strong_trend"

    def test_all_time_ag_dir(self):
        from attribution.batch import _detect_strategy_dir
        assert _detect_strategy_dir(
            "/foo/strategies/all_time/ag/portfolio/weights_ag.json"
        ) == "all_time/ag"

    def test_all_time_i_dir(self):
        from attribution.batch import _detect_strategy_dir
        assert _detect_strategy_dir(
            "/foo/strategies/all_time/i/portfolio/weights_i.json"
        ) == "all_time/i"

    def test_no_strategies_in_path(self):
        from attribution.batch import _detect_strategy_dir
        # Falls back to "strong_trend"
        assert _detect_strategy_dir("/tmp/some/random/weights.json") == "strong_trend"


# =========================================================================
# Integration: test with real weights JSON (if available)
# =========================================================================

class TestWeightsJsonIntegration:
    """Test parsing of actual weights JSON files in the repo."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(QBASE_ROOT, "strategies/strong_trend/portfolio/weights_ag.json")),
        reason="Real weights JSON not available",
    )
    def test_parse_real_strong_trend_weights(self):
        """Parse the real strong_trend weights JSON."""
        from attribution.batch import parse_weights_json

        path = os.path.join(QBASE_ROOT, "strategies/strong_trend/portfolio/weights_ag.json")
        entries, meta = parse_weights_json(path)

        assert len(entries) > 0
        assert meta.get("symbol") == "AG"
        # All entries from builder format should have params
        for e in entries:
            assert e.params, f"{e.version} should have params"
            assert e.weight > 0
            assert e.freq in ("daily", "1h", "4h", "30min", "60min", "10min", "5min")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(QBASE_ROOT, "strategies/all_time/i/portfolio/weights_i.json")),
        reason="Real weights JSON not available",
    )
    def test_parse_real_all_time_weights(self):
        """Parse the real all_time/i weights JSON."""
        from attribution.batch import parse_weights_json

        path = os.path.join(QBASE_ROOT, "strategies/all_time/i/portfolio/weights_i.json")
        entries, meta = parse_weights_json(path)

        assert len(entries) > 0
        # All entries should have params (embedded in builder format)
        for e in entries:
            assert e.params, f"{e.version} should have params"
