"""Tests for attribution analysis modules."""
import sys
import importlib
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
import pytest

# Strategy class names mapping (mirrors strong_trend/optimizer.py)
_STRATEGY_CLASSES = {f"v{i}": f"StrongTrendV{i}" for i in range(1, 51)}
_STRATEGY_CLASSES["v3"] = "DonchianADXChandelierStrategy"


def _load_strategy_class(version: str):
    """Load strong_trend strategy class without importing the optimizer module."""
    mod = importlib.import_module(f"strategies.strong_trend.{version}")
    return getattr(mod, _STRATEGY_CLASSES[version])


class TestRunBacktestFull:
    """Test the full-result backtest helper."""

    def test_returns_result_with_trades(self):
        """run_backtest_full should return a BacktestResult with trades DataFrame."""
        from attribution.signal import run_backtest_full
        from strategies.optimizer_core import create_strategy_with_params

        strategy_cls = _load_strategy_class("v12")
        params = {
            "aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29,
            "vol_mom_period": 25, "atr_trail_mult": 4.814,
        }
        strategy = create_strategy_with_params(strategy_cls, params)
        result = run_backtest_full(strategy, "AG", "2025-01-01", "2026-03-01", freq="daily")

        assert result is not None
        assert hasattr(result, 'sharpe')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'equity_curve')
        assert result.trades is not None
        assert len(result.trades) > 0

    def test_returns_none_on_bad_symbol(self):
        """run_backtest_full should return None for invalid input."""
        from attribution.signal import run_backtest_full
        from strategies.optimizer_core import create_strategy_with_params

        strategy_cls = _load_strategy_class("v12")
        strategy = create_strategy_with_params(strategy_cls, {})
        result = run_backtest_full(strategy, "INVALID_SYMBOL", "2025-01-01", "2026-03-01")
        assert result is None


class TestSignalAttribution:
    """Test signal ablation engine."""

    def test_ablation_produces_different_sharpes(self):
        """Ablating an important indicator should change the Sharpe."""
        from attribution.signal import run_signal_attribution

        strategy_cls = _load_strategy_class("v12")
        params = {
            "aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29,
            "vol_mom_period": 25, "atr_trail_mult": 4.814,
        }
        result = run_signal_attribution(
            strategy_cls, params, "AG", "2025-01-01", "2026-03-01", freq="daily",
        )
        assert result.baseline_sharpe > 0
        assert len(result.contributions) == 3
        max_contrib = max(c['contribution'] for c in result.contributions.values())
        assert max_contrib > 0.1, "At least one indicator should matter"
        assert result.dominant_indicator != ""

    def test_auto_discovery_without_config(self):
        """Auto-discovery should find indicator arrays when no INDICATOR_CONFIG."""
        from attribution.signal import _discover_indicator_arrays, run_backtest_full
        from strategies.optimizer_core import create_strategy_with_params

        strategy_cls = _load_strategy_class("v12")
        params = {
            "aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29,
            "vol_mom_period": 25, "atr_trail_mult": 4.814,
        }
        strategy = create_strategy_with_params(strategy_cls, params)
        run_backtest_full(strategy, "AG", "2025-01-01", "2026-03-01", freq="daily")

        discovered = _discover_indicator_arrays(strategy)
        array_names = [d['array_attr'] for d in discovered]
        assert '_atr' not in array_names, "_atr (stop-loss) should be excluded"
        assert len(discovered) >= 3, "Should discover at least 3 indicator arrays"


class TestTradePairing:
    """Test trade pairing logic."""

    def test_simple_long_roundtrip(self):
        """A buy followed by sell should produce one round-trip."""
        from attribution.regime import pair_trades

        trades_df = pd.DataFrame([
            {'datetime': '2025-01-10', 'side': 'buy', 'lots': 5, 'price': 100.0},
            {'datetime': '2025-01-20', 'side': 'sell', 'lots': 5, 'price': 110.0},
        ])
        pairs = pair_trades(trades_df)
        assert len(pairs) == 1
        assert pairs[0]['side'] == 1
        assert pairs[0]['entry_price'] == 100.0
        assert pairs[0]['exit_price'] == 110.0
        assert pairs[0]['pnl_pct'] == pytest.approx(0.10, rel=0.01)

    def test_partial_close(self):
        """Partial close should produce two round-trips."""
        from attribution.regime import pair_trades

        trades_df = pd.DataFrame([
            {'datetime': '2025-01-10', 'side': 'buy', 'lots': 10, 'price': 100.0},
            {'datetime': '2025-01-15', 'side': 'sell', 'lots': 5, 'price': 105.0},
            {'datetime': '2025-01-20', 'side': 'sell', 'lots': 5, 'price': 110.0},
        ])
        pairs = pair_trades(trades_df)
        assert len(pairs) == 2
        assert pairs[0]['lots'] == 5
        assert pairs[1]['lots'] == 5

    def test_empty_trades(self):
        """Empty DataFrame should return empty list."""
        from attribution.regime import pair_trades

        trades_df = pd.DataFrame(columns=['datetime', 'side', 'lots', 'price'])
        pairs = pair_trades(trades_df)
        assert pairs == []


class TestRegimeAttribution:
    """Test regime attribution on real strategy."""

    def test_produces_regime_stats(self):
        """Regime attribution should produce stats for all three dimensions."""
        from attribution.regime import run_regime_attribution

        strategy_cls = _load_strategy_class("v12")
        params = {
            "aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29,
            "vol_mom_period": 25, "atr_trail_mult": 4.814,
        }
        result = run_regime_attribution(
            strategy_cls, params, "AG", "2025-01-01", "2026-03-01", freq="daily",
        )
        assert result.total_trades > 0
        assert len(result.by_trend) > 0
        assert len(result.by_volatility) > 0
        assert len(result.by_activity) > 0
        assert result.best_regime != ""
        assert result.worst_regime != ""
