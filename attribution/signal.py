"""Signal attribution via ablation testing."""
import sys
from pathlib import Path
from dataclasses import dataclass, field

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SignalAttributionResult:
    strategy_version: str
    symbol: str
    period: str
    baseline_sharpe: float
    baseline_trades: int
    contributions: dict = field(default_factory=dict)
    dominant_indicator: str = ""
    redundant_indicators: list = field(default_factory=list)


def run_backtest_full(strategy, symbol, start, end, freq="daily", data_dir=None):
    """Run backtest and return the full BacktestResult (with trades DataFrame).

    Unlike optimizer_core.run_single_backtest which returns a summary dict,
    this returns the raw BacktestResult object from AlphaForge.

    Returns:
        BacktestResult or None on failure.
    """
    try:
        from alphaforge.data.market import MarketDataLoader
        from alphaforge.data.contract_specs import ContractSpecManager
        from alphaforge.engine.event_driven import EventDrivenBacktester
        from strategies.optimizer_core import map_freq, resample_bars

        if data_dir is None:
            from config import get_data_dir
            data_dir = get_data_dir()

        loader = MarketDataLoader(data_dir)
        load_freq, resample_factor = map_freq(freq)
        bars = loader.load(symbol, freq=load_freq, start=start, end=end)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None

        if resample_factor > 1:
            bars = resample_bars(bars, resample_factor)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None

        engine = EventDrivenBacktester(
            spec_manager=ContractSpecManager(),
            initial_capital=1_000_000,
            slippage_ticks=1.0,
        )
        result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)
        return result
    except Exception:
        return None
