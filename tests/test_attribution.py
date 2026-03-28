"""Tests for attribution analysis modules."""
import sys
import importlib
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
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
