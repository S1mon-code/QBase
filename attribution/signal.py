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


def _discover_indicator_arrays(strategy) -> list[dict]:
    """Auto-discover indicator arrays from a strategy instance.

    Scans for numpy array attributes with '_' prefix, excluding known
    non-indicator arrays (_atr is used for stop-loss, not signals).
    """
    EXCLUDE = {'_atr', '_avg_volume', '_4h_map', '_tradeable_mask'}
    discovered = []
    for attr_name in dir(strategy):
        if not attr_name.startswith('_') or attr_name.startswith('__'):
            continue
        if attr_name in EXCLUDE:
            continue
        val = getattr(strategy, attr_name, None)
        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 0:
            median_val = float(np.nanmedian(val))
            discovered.append({
                'name': attr_name.lstrip('_').replace('_', ' ').title(),
                'array_attr': attr_name,
                'neutral_value': median_val,
                'role': 'unknown',
            })
    return discovered


def run_signal_attribution(
    strategy_cls,
    params: dict,
    symbol: str,
    start: str,
    end: str,
    freq: str = "daily",
    indicator_config: list[dict] | None = None,
    data_dir: str | None = None,
) -> SignalAttributionResult:
    """Run ablation test for each indicator in the strategy.

    For each indicator, replaces its precomputed array with a neutral value
    and re-runs the backtest. The difference in Sharpe measures that
    indicator's contribution.
    """
    from strategies.optimizer_core import create_strategy_with_params

    # 1. Baseline run
    baseline_strategy = create_strategy_with_params(strategy_cls, params)
    baseline_result = run_backtest_full(baseline_strategy, symbol, start, end, freq, data_dir)
    if baseline_result is None:
        return SignalAttributionResult(
            strategy_version=getattr(strategy_cls, 'name', str(strategy_cls)),
            symbol=symbol, period=f"{start} ~ {end}",
            baseline_sharpe=-999.0, baseline_trades=0,
        )

    baseline_sharpe = float(baseline_result.sharpe)
    baseline_trades = int(baseline_result.n_trades)

    # 2. Determine indicator config
    if indicator_config is None:
        indicator_config = getattr(strategy_cls, 'INDICATOR_CONFIG', None)
    if indicator_config is None:
        indicator_config = _discover_indicator_arrays(baseline_strategy)

    if len(indicator_config) <= 1:
        return SignalAttributionResult(
            strategy_version=getattr(strategy_cls, 'name', str(strategy_cls)),
            symbol=symbol, period=f"{start} ~ {end}",
            baseline_sharpe=baseline_sharpe, baseline_trades=baseline_trades,
            dominant_indicator=indicator_config[0]['name'] if indicator_config else "",
        )

    # 3. Ablation: for each indicator, neutralize and re-run
    contributions = {}
    for indicator in indicator_config:
        # Create fresh strategy, run backtest to populate arrays
        ablated_strategy = create_strategy_with_params(strategy_cls, params)
        ablated_result_temp = run_backtest_full(
            ablated_strategy, symbol, start, end, freq, data_dir,
        )
        if ablated_result_temp is None:
            continue

        # Replace the target array with neutral value
        original_array = getattr(ablated_strategy, indicator['array_attr'], None)
        if original_array is None or not isinstance(original_array, np.ndarray):
            continue

        neutral_array = np.full_like(original_array, indicator['neutral_value'])
        setattr(ablated_strategy, indicator['array_attr'], neutral_array)

        # Patch on_init_arrays to no-op so our modified array survives
        ablated_strategy.on_init_arrays = lambda ctx, bars: None

        ablated_result = run_backtest_full(
            ablated_strategy, symbol, start, end, freq, data_dir,
        )
        if ablated_result is None:
            ablated_sharpe = -999.0
        else:
            ablated_sharpe = float(ablated_result.sharpe)
            if np.isnan(ablated_sharpe) or np.isinf(ablated_sharpe):
                ablated_sharpe = -999.0

        contribution = baseline_sharpe - ablated_sharpe
        if ablated_sharpe <= -900:
            contribution = baseline_sharpe

        contributions[indicator['name']] = {
            'ablated_sharpe': round(ablated_sharpe, 3),
            'contribution': round(contribution, 3),
            'pct_contribution': round(
                contribution / max(abs(baseline_sharpe), 0.01) * 100, 1
            ),
            'role': indicator.get('role', 'unknown'),
        }

    # 4. Determine dominant and redundant
    dominant = ""
    redundant = []
    if contributions:
        dominant = max(contributions, key=lambda k: contributions[k]['contribution'])
        for name, c in contributions.items():
            if abs(c['pct_contribution']) < 5.0:
                redundant.append(name)

    return SignalAttributionResult(
        strategy_version=getattr(strategy_cls, 'name', str(strategy_cls)),
        symbol=symbol,
        period=f"{start} ~ {end}",
        baseline_sharpe=round(baseline_sharpe, 3),
        baseline_trades=baseline_trades,
        contributions=contributions,
        dominant_indicator=dominant,
        redundant_indicators=redundant,
    )
