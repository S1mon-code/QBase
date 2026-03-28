# Attribution Analysis Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated signal attribution (ablation test) and regime attribution (per-regime trade stats) to QBase's pipeline as Step 4.5.

**Architecture:** Three new modules in `attribution/` — `signal.py` (ablation engine), `regime.py` (regime tagging + stats), `report.py` (Markdown generator). Integrates into `validate_and_iterate.py` after test-set validation. Uses existing `optimizer_core.run_single_backtest` infrastructure.

**Tech Stack:** Python, NumPy, existing QBase indicators, AlphaForge backtest engine.

**Spec:** `docs/superpowers/specs/2026-03-28-attribution-analysis-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `attribution/__init__.py` | Package init, public API exports |
| Create | `attribution/signal.py` | Signal ablation engine + `SignalAttributionResult` |
| Create | `attribution/regime.py` | Regime tagging, trade pairing, per-regime stats + `RegimeAttributionResult` |
| Create | `attribution/report.py` | Markdown report generation + `run_full_attribution` orchestrator |
| Create | `tests/test_attribution.py` | Unit tests for all three modules |
| Modify | `strategies/strong_trend/v12.py` | Add `INDICATOR_CONFIG` class attribute |
| Modify | `strategies/strong_trend/validate_and_iterate.py` | Wire in attribution after validation |
| Modify | `CLAUDE.md` | Add Step 4.5 documentation |

---

### Task 1: Backtest Helper — Full Result Runner

**Files:**
- Create: `attribution/__init__.py`
- Create: `attribution/signal.py` (partial — just the backtest helper)
- Create: `tests/test_attribution.py` (partial)

The existing `optimizer_core.run_single_backtest` returns a summary dict (sharpe, max_drawdown, etc.) but drops the `BacktestResult` object which has `trades` DataFrame and `equity_curve`. We need a helper that returns the full result.

- [ ] **Step 1: Create attribution package with backtest helper**

Create `attribution/__init__.py`:

```python
"""Attribution analysis for QBase strategies."""
```

Create the core of `attribution/signal.py` with just the backtest helper:

```python
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
```

- [ ] **Step 2: Write test for backtest helper**

Create `tests/test_attribution.py`:

```python
"""Tests for attribution analysis modules."""
import sys
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pytest


class TestRunBacktestFull:
    """Test the full-result backtest helper."""

    def test_returns_result_with_trades(self):
        """run_backtest_full should return a BacktestResult with trades DataFrame."""
        from attribution.signal import run_backtest_full
        from strategies.optimizer_core import create_strategy_with_params
        from strategies.strong_trend.optimizer import load_strategy_class

        strategy_cls = load_strategy_class("v12")
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
        from strategies.strong_trend.optimizer import load_strategy_class

        strategy_cls = load_strategy_class("v12")
        strategy = create_strategy_with_params(strategy_cls, {})
        result = run_backtest_full(strategy, "INVALID_SYMBOL", "2025-01-01", "2026-03-01")
        assert result is None
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestRunBacktestFull -v`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/simon/Desktop/QBase
git add attribution/__init__.py attribution/signal.py tests/test_attribution.py
git commit -m "[attribution] add backtest helper returning full BacktestResult"
```

---

### Task 2: Signal Attribution — Ablation Engine

**Files:**
- Modify: `attribution/signal.py`
- Modify: `tests/test_attribution.py`
- Modify: `strategies/strong_trend/v12.py` (add INDICATOR_CONFIG)

- [ ] **Step 1: Add INDICATOR_CONFIG to v12**

Add this class attribute inside `StrongTrendV12` in `strategies/strong_trend/v12.py`, right after `contract_multiplier`:

```python
    INDICATOR_CONFIG = [
        {'name': 'Aroon Oscillator', 'array_attr': '_aroon_osc', 'neutral_value': 100, 'role': 'trend'},
        {'name': 'PPO Histogram', 'array_attr': '_ppo_hist', 'neutral_value': 1.0, 'role': 'momentum'},
        {'name': 'Volume Momentum', 'array_attr': '_vol_mom', 'neutral_value': 2.0, 'role': 'volume'},
    ]
```

- [ ] **Step 2: Write failing test for signal attribution**

Add to `tests/test_attribution.py`:

```python
class TestSignalAttribution:
    """Test signal ablation engine."""

    def test_ablation_produces_different_sharpes(self):
        """Ablating an important indicator should change the Sharpe."""
        from attribution.signal import run_signal_attribution
        from strategies.strong_trend.optimizer import load_strategy_class

        strategy_cls = load_strategy_class("v12")
        params = {
            "aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29,
            "vol_mom_period": 25, "atr_trail_mult": 4.814,
        }
        result = run_signal_attribution(
            strategy_cls, params, "AG", "2025-01-01", "2026-03-01", freq="daily",
        )
        assert result.baseline_sharpe > 0
        assert len(result.contributions) == 3  # 3 indicators in v12
        # At least one indicator should have meaningful contribution
        max_contrib = max(c['contribution'] for c in result.contributions.values())
        assert max_contrib > 0.1, "At least one indicator should matter"
        assert result.dominant_indicator != ""

    def test_auto_discovery_without_config(self):
        """Auto-discovery should find indicator arrays when no INDICATOR_CONFIG."""
        from attribution.signal import _discover_indicator_arrays
        from strategies.optimizer_core import create_strategy_with_params
        from strategies.strong_trend.optimizer import load_strategy_class

        strategy_cls = load_strategy_class("v12")
        params = {
            "aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29,
            "vol_mom_period": 25, "atr_trail_mult": 4.814,
        }
        strategy = create_strategy_with_params(strategy_cls, params)
        # Need to populate arrays first via a backtest run
        from attribution.signal import run_backtest_full
        run_backtest_full(strategy, "AG", "2025-01-01", "2026-03-01", freq="daily")

        discovered = _discover_indicator_arrays(strategy)
        # Should find arrays like _aroon_osc, _ppo_hist, _vol_mom but NOT _atr
        array_names = [d['array_attr'] for d in discovered]
        assert '_atr' not in array_names, "_atr (stop-loss) should be excluded"
        assert len(discovered) >= 3, "Should discover at least 3 indicator arrays"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestSignalAttribution -v`
Expected: FAIL (functions not implemented yet)

- [ ] **Step 4: Implement run_signal_attribution and _discover_indicator_arrays**

Add to `attribution/signal.py`:

```python
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
        # Auto-discover from the baseline strategy (arrays are populated)
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
        ablated_strategy = create_strategy_with_params(strategy_cls, params)

        # Run on_init_arrays to populate all arrays, then replace one
        ablated_result_temp = run_backtest_full(
            ablated_strategy, symbol, start, end, freq, data_dir,
        )
        if ablated_result_temp is None:
            continue

        # Now ablated_strategy has all arrays populated.
        # Replace the target array with neutral value.
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
        # Cap contribution: if ablation caused total failure, contribution = baseline
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestSignalAttribution -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/simon/Desktop/QBase
git add attribution/signal.py strategies/strong_trend/v12.py tests/test_attribution.py
git commit -m "[attribution] implement signal ablation engine with auto-discovery"
```

---

### Task 3: Regime Attribution — Trade Pairing + Regime Tagging

**Files:**
- Create: `attribution/regime.py`
- Modify: `tests/test_attribution.py`

- [ ] **Step 1: Write failing tests for trade pairing and regime attribution**

Add to `tests/test_attribution.py`:

```python
import pandas as pd


class TestTradePairing:
    """Test trade pairing logic."""

    def test_simple_long_roundtrip(self):
        """A buy followed by close_long should produce one round-trip."""
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
        from strategies.strong_trend.optimizer import load_strategy_class

        strategy_cls = load_strategy_class("v12")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestTradePairing tests/test_attribution.py::TestRegimeAttribution -v`
Expected: FAIL

- [ ] **Step 3: Implement regime.py**

Create `attribution/regime.py`:

```python
"""Regime attribution — tag trades with market regime and compute per-regime stats."""
import sys
from pathlib import Path
from dataclasses import dataclass, field

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@dataclass
class RegimeStats:
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0
    avg_holding_bars: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


@dataclass
class RegimeAttributionResult:
    strategy_version: str
    symbol: str
    period: str
    total_trades: int
    total_sharpe: float
    by_trend: dict = field(default_factory=dict)
    by_volatility: dict = field(default_factory=dict)
    by_activity: dict = field(default_factory=dict)
    cross_trend_vol: dict = field(default_factory=dict)
    best_regime: str = ""
    worst_regime: str = ""


def pair_trades(trades_df: pd.DataFrame) -> list[dict]:
    """Pair entry and exit trades into round-trip records.

    Walks through trades chronologically, matching opens (buy) with closes (sell)
    for long trades. Handles partial closes by splitting into sub-trades.
    """
    if trades_df is None or len(trades_df) == 0:
        return []

    pairs = []
    open_lots = 0
    open_price = 0.0
    open_dt = None
    open_side = 0  # 1=long, -1=short

    for _, row in trades_df.iterrows():
        side_str = str(row.get('side', ''))
        lots = int(row.get('lots', 0))
        price = float(row.get('price', 0))
        dt = row.get('datetime', '')

        if side_str == 'buy':
            if open_lots == 0:
                # New long entry
                open_lots = lots
                open_price = price
                open_dt = dt
                open_side = 1
            elif open_side == 1:
                # Add to existing long
                total_cost = open_price * open_lots + price * lots
                open_lots += lots
                open_price = total_cost / open_lots if open_lots > 0 else price
            elif open_side == -1:
                # Close short
                close_lots = min(lots, open_lots)
                pnl_pct = (open_price - price) / open_price if open_price > 0 else 0
                pairs.append({
                    'entry_datetime': open_dt,
                    'exit_datetime': dt,
                    'side': -1,
                    'entry_price': open_price,
                    'exit_price': price,
                    'lots': close_lots,
                    'pnl_pct': pnl_pct,
                    'holding_bars': 0,  # Will be filled by caller if needed
                })
                open_lots -= close_lots
                if open_lots <= 0:
                    open_lots = 0
                    open_side = 0

        elif side_str == 'sell':
            if open_lots > 0 and open_side == 1:
                # Close long (full or partial)
                close_lots = min(lots, open_lots)
                pnl_pct = (price - open_price) / open_price if open_price > 0 else 0
                pairs.append({
                    'entry_datetime': open_dt,
                    'exit_datetime': dt,
                    'side': 1,
                    'entry_price': open_price,
                    'exit_price': price,
                    'lots': close_lots,
                    'pnl_pct': pnl_pct,
                    'holding_bars': 0,
                })
                open_lots -= close_lots
                if open_lots <= 0:
                    open_lots = 0
                    open_side = 0
            elif open_lots == 0:
                # New short entry
                open_lots = lots
                open_price = price
                open_dt = dt
                open_side = -1

    return pairs


def _rolling_percentile(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling percentile rank of each value within its trailing window."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        w = arr[i - window:i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) < 2:
            continue
        rank = np.sum(valid < arr[i]) / (len(valid) - 1)
        out[i] = rank
    return out


def _compute_regime_labels(highs, lows, closes, volumes):
    """Compute regime labels for every bar across three dimensions."""
    from indicators.trend.adx import adx
    from indicators.volatility.atr import atr
    from indicators.trend.sma import sma

    n = len(closes)
    adx_arr = adx(highs, lows, closes, period=14)
    atr_arr = atr(highs, lows, closes, period=14)
    vol_ma = sma(volumes.astype(np.float64), 20)

    # Trend regime: strong / weak / none
    trend_labels = np.full(n, 'unknown', dtype=object)
    for i in range(n):
        if np.isnan(adx_arr[i]):
            trend_labels[i] = 'unknown'
        elif adx_arr[i] > 25:
            trend_labels[i] = 'strong'
        elif adx_arr[i] > 15:
            trend_labels[i] = 'weak'
        else:
            trend_labels[i] = 'none'

    # Volatility regime: high / normal / low
    atr_pctile = _rolling_percentile(atr_arr, 252)
    vol_labels = np.full(n, 'unknown', dtype=object)
    for i in range(n):
        if np.isnan(atr_pctile[i]):
            vol_labels[i] = 'unknown'
        elif atr_pctile[i] > 0.75:
            vol_labels[i] = 'high'
        elif atr_pctile[i] > 0.25:
            vol_labels[i] = 'normal'
        else:
            vol_labels[i] = 'low'

    # Activity regime: active / normal / quiet
    activity_labels = np.full(n, 'unknown', dtype=object)
    for i in range(n):
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            activity_labels[i] = 'unknown'
        else:
            ratio = volumes[i] / vol_ma[i]
            if ratio > 1.5:
                activity_labels[i] = 'active'
            elif ratio > 0.7:
                activity_labels[i] = 'normal'
            else:
                activity_labels[i] = 'quiet'

    return trend_labels, vol_labels, activity_labels


def _compute_regime_stats(trades_in_regime: list[dict]) -> RegimeStats:
    """Compute stats for a group of trades."""
    if not trades_in_regime:
        return RegimeStats()

    pnls = [t['pnl_pct'] for t in trades_in_regime]
    wins = [p for p in pnls if p > 0]
    holdings = [t.get('holding_bars', 0) for t in trades_in_regime]

    return RegimeStats(
        n_trades=len(pnls),
        win_rate=round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        avg_pnl_pct=round(float(np.mean(pnls)) * 100, 2) if pnls else 0,
        total_pnl_pct=round(float(np.sum(pnls)) * 100, 2) if pnls else 0,
        avg_holding_bars=round(float(np.mean(holdings)), 1) if holdings else 0,
        best_trade_pnl=round(float(max(pnls)) * 100, 2) if pnls else 0,
        worst_trade_pnl=round(float(min(pnls)) * 100, 2) if pnls else 0,
    )


def _match_datetime_to_bar_index(dt, bar_datetimes):
    """Find the bar index closest to (and not after) the given datetime."""
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    indices = np.where(bar_datetimes <= dt)[0]
    if len(indices) == 0:
        return 0
    return indices[-1]


def run_regime_attribution(
    strategy_cls,
    params: dict,
    symbol: str,
    start: str,
    end: str,
    freq: str = "daily",
    data_dir: str | None = None,
) -> RegimeAttributionResult:
    """Run strategy, tag each trade with market regime, compute per-regime stats."""
    from strategies.optimizer_core import create_strategy_with_params
    from attribution.signal import run_backtest_full
    from alphaforge.data.market import MarketDataLoader
    from strategies.optimizer_core import map_freq, resample_bars

    version = getattr(strategy_cls, 'name', str(strategy_cls))

    # 1. Run backtest
    strategy = create_strategy_with_params(strategy_cls, params)
    result = run_backtest_full(strategy, symbol, start, end, freq, data_dir)
    if result is None or result.trades is None or len(result.trades) == 0:
        return RegimeAttributionResult(
            strategy_version=version, symbol=symbol,
            period=f"{start} ~ {end}", total_trades=0, total_sharpe=-999,
        )

    # 2. Load bar data for regime computation
    if data_dir is None:
        from config import get_data_dir
        data_dir = get_data_dir()
    loader = MarketDataLoader(data_dir)
    load_freq, resample_factor = map_freq(freq)
    bars = loader.load(symbol, freq=load_freq, start=start, end=end)
    if resample_factor > 1:
        bars = resample_bars(bars, resample_factor)

    closes = bars._close
    highs = bars._high
    lows = bars._low
    volumes = bars._volume
    bar_datetimes = bars._datetime if hasattr(bars, '_datetime') else np.arange(len(closes))

    # 3. Compute regime labels
    trend_labels, vol_labels, activity_labels = _compute_regime_labels(
        highs, lows, closes, volumes,
    )

    # 4. Pair trades
    paired = pair_trades(result.trades)
    if not paired:
        return RegimeAttributionResult(
            strategy_version=version, symbol=symbol,
            period=f"{start} ~ {end}", total_trades=0,
            total_sharpe=float(result.sharpe),
        )

    # 5. Tag each trade with regime at entry
    for trade in paired:
        idx = _match_datetime_to_bar_index(trade['entry_datetime'], bar_datetimes)
        trade['trend_regime'] = str(trend_labels[idx])
        trade['vol_regime'] = str(vol_labels[idx])
        trade['activity_regime'] = str(activity_labels[idx])

    # 6. Group and compute stats
    by_trend = {}
    by_vol = {}
    by_act = {}
    cross = {}

    for label in ['strong', 'weak', 'none', 'unknown']:
        group = [t for t in paired if t.get('trend_regime') == label]
        if group:
            by_trend[label] = _compute_regime_stats(group)

    for label in ['high', 'normal', 'low', 'unknown']:
        group = [t for t in paired if t.get('vol_regime') == label]
        if group:
            by_vol[label] = _compute_regime_stats(group)

    for label in ['active', 'normal', 'quiet', 'unknown']:
        group = [t for t in paired if t.get('activity_regime') == label]
        if group:
            by_act[label] = _compute_regime_stats(group)

    # Cross: trend x volatility
    for tl in ['strong', 'weak', 'none']:
        for vl in ['high', 'normal', 'low']:
            group = [t for t in paired
                     if t.get('trend_regime') == tl and t.get('vol_regime') == vl]
            if group:
                cross[(tl, vl)] = _compute_regime_stats(group)

    # 7. Best/worst regime (by average PnL, min 2 trades)
    all_regimes = {}
    for tl, stats in by_trend.items():
        if stats.n_trades >= 2:
            all_regimes[f"trend={tl}"] = stats.avg_pnl_pct
    for vl, stats in by_vol.items():
        if stats.n_trades >= 2:
            all_regimes[f"vol={vl}"] = stats.avg_pnl_pct
    for al, stats in by_act.items():
        if stats.n_trades >= 2:
            all_regimes[f"activity={al}"] = stats.avg_pnl_pct

    best = max(all_regimes, key=all_regimes.get) if all_regimes else ""
    worst = min(all_regimes, key=all_regimes.get) if all_regimes else ""

    return RegimeAttributionResult(
        strategy_version=version,
        symbol=symbol,
        period=f"{start} ~ {end}",
        total_trades=len(paired),
        total_sharpe=round(float(result.sharpe), 3),
        by_trend=by_trend,
        by_volatility=by_vol,
        by_activity=by_act,
        cross_trend_vol=cross,
        best_regime=best,
        worst_regime=worst,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestTradePairing tests/test_attribution.py::TestRegimeAttribution -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/simon/Desktop/QBase
git add attribution/regime.py tests/test_attribution.py
git commit -m "[attribution] implement regime attribution with trade pairing"
```

---

### Task 4: Report Generator

**Files:**
- Create: `attribution/report.py`
- Modify: `tests/test_attribution.py`

- [ ] **Step 1: Write failing test for report generation**

Add to `tests/test_attribution.py`:

```python
import os


class TestReportGenerator:
    """Test Markdown report generation."""

    def test_generates_report_file(self, tmp_path):
        """Should generate a Markdown file with all required sections."""
        from attribution.signal import SignalAttributionResult
        from attribution.regime import RegimeAttributionResult, RegimeStats
        from attribution.report import generate_attribution_report

        signal = SignalAttributionResult(
            strategy_version="strong_trend_v12",
            symbol="AG",
            period="2025-01-01 ~ 2026-03-01",
            baseline_sharpe=3.09,
            baseline_trades=18,
            contributions={
                'Aroon Oscillator': {'ablated_sharpe': 1.2, 'contribution': 1.89, 'pct_contribution': 61.2, 'role': 'trend'},
                'PPO Histogram': {'ablated_sharpe': 2.1, 'contribution': 0.99, 'pct_contribution': 32.0, 'role': 'momentum'},
                'Volume Momentum': {'ablated_sharpe': 2.9, 'contribution': 0.19, 'pct_contribution': 6.1, 'role': 'volume'},
            },
            dominant_indicator='Aroon Oscillator',
            redundant_indicators=[],
        )

        regime = RegimeAttributionResult(
            strategy_version="strong_trend_v12",
            symbol="AG",
            period="2025-01-01 ~ 2026-03-01",
            total_trades=18,
            total_sharpe=3.09,
            by_trend={'strong': RegimeStats(n_trades=12, win_rate=75.0, avg_pnl_pct=2.5, total_pnl_pct=30.0, avg_holding_bars=15, best_trade_pnl=8.0, worst_trade_pnl=-1.5)},
            by_volatility={'high': RegimeStats(n_trades=10, win_rate=70.0, avg_pnl_pct=2.0, total_pnl_pct=20.0, avg_holding_bars=12, best_trade_pnl=7.0, worst_trade_pnl=-2.0)},
            by_activity={'active': RegimeStats(n_trades=8, win_rate=62.5, avg_pnl_pct=1.5, total_pnl_pct=12.0, avg_holding_bars=10, best_trade_pnl=5.0, worst_trade_pnl=-3.0)},
            cross_trend_vol={('strong', 'high'): RegimeStats(n_trades=8, win_rate=75.0, avg_pnl_pct=3.0, total_pnl_pct=24.0, avg_holding_bars=14, best_trade_pnl=8.0, worst_trade_pnl=-1.0)},
            best_regime="trend=strong",
            worst_regime="activity=quiet",
        )

        output = str(tmp_path / "test_report.md")
        result_path = generate_attribution_report(signal, regime, output)

        assert os.path.exists(result_path)
        content = open(result_path).read()
        assert "Signal Attribution" in content
        assert "Regime Attribution" in content
        assert "Aroon Oscillator" in content
        assert "Dominant indicator" in content
        assert "Trend Strength" in content
        assert "Cross Analysis" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestReportGenerator -v`
Expected: FAIL

- [ ] **Step 3: Implement report.py**

Create `attribution/report.py`:

```python
"""Attribution report generator — Markdown reports + orchestrator."""
import os
from datetime import datetime


def generate_attribution_report(signal_result, regime_result, output_path: str) -> str:
    """Generate Markdown attribution report combining signal and regime results."""
    lines = []

    version = signal_result.strategy_version
    symbol = signal_result.symbol
    period = signal_result.period

    lines.append(f"# {version} Attribution Report — {symbol} ({period})")
    lines.append("")

    # === Signal Attribution ===
    lines.append("## Signal Attribution (Ablation Test)")
    lines.append("")
    lines.append(f"**Baseline Sharpe: {signal_result.baseline_sharpe:.3f}** | "
                 f"Trades: {signal_result.baseline_trades}")
    lines.append("")

    if signal_result.contributions:
        lines.append("| Indicator | Role | Without It | Contribution | % of Total |")
        lines.append("|-----------|------|:----------:|:------------:|:----------:|")
        for name, c in sorted(signal_result.contributions.items(),
                               key=lambda x: -x[1]['contribution']):
            lines.append(
                f"| {name} | {c['role']} | "
                f"{c['ablated_sharpe']:.3f} | "
                f"{c['contribution']:+.3f} | "
                f"{c['pct_contribution']:.1f}% |"
            )
        lines.append("")
        lines.append(f"**Dominant indicator**: {signal_result.dominant_indicator}")
        if signal_result.redundant_indicators:
            lines.append(f"**Redundant indicators**: {', '.join(signal_result.redundant_indicators)}")
        else:
            lines.append("**Redundant indicators**: None")
    else:
        lines.append("*No ablation data available.*")

    # Auto insights
    lines.append("")
    lines.append("### Interpretation")
    if signal_result.contributions:
        for name, c in signal_result.contributions.items():
            if c['pct_contribution'] > 60:
                lines.append(f"- Strategy is heavily dependent on **{name}** "
                             f"({c['pct_contribution']:.0f}% contribution). "
                             f"Consider if this is a feature or a risk.")
            elif c['pct_contribution'] < 5:
                lines.append(f"- **{name}** adds minimal value ({c['pct_contribution']:.1f}%). "
                             f"Consider removing for simplicity.")
    lines.append("")

    # === Regime Attribution ===
    lines.append("## Regime Attribution")
    lines.append("")
    lines.append(f"Total round-trip trades analyzed: {regime_result.total_trades}")
    lines.append("")

    # By Trend
    _write_regime_table(lines, "By Trend Strength (ADX)", regime_result.by_trend,
                        {'strong': 'Strong (ADX>25)', 'weak': 'Weak (15-25)',
                         'none': 'None (<15)', 'unknown': 'Unknown'})

    # By Volatility
    _write_regime_table(lines, "By Volatility (ATR Percentile)", regime_result.by_volatility,
                        {'high': 'High (>75th)', 'normal': 'Normal (25-75th)',
                         'low': 'Low (<25th)', 'unknown': 'Unknown'})

    # By Activity
    _write_regime_table(lines, "By Volume Activity", regime_result.by_activity,
                        {'active': 'Active (>1.5x)', 'normal': 'Normal (0.7-1.5x)',
                         'quiet': 'Quiet (<0.7x)', 'unknown': 'Unknown'})

    # Cross analysis
    if regime_result.cross_trend_vol:
        lines.append("### Cross Analysis: Trend x Volatility")
        lines.append("")
        lines.append("| | High Vol | Normal Vol | Low Vol |")
        lines.append("|--|:--:|:--:|:--:|")
        for tl in ['strong', 'weak', 'none']:
            label = {'strong': '**Strong Trend**', 'weak': '**Weak Trend**', 'none': '**No Trend**'}[tl]
            cells = []
            for vl in ['high', 'normal', 'low']:
                stats = regime_result.cross_trend_vol.get((tl, vl))
                if stats and stats.n_trades > 0:
                    cells.append(f"{stats.win_rate:.0f}% ({stats.n_trades})")
                else:
                    cells.append("—")
            lines.append(f"| {label} | {' | '.join(cells)} |")
        lines.append("")

    # Best/worst
    if regime_result.best_regime:
        lines.append(f"**Best regime**: {regime_result.best_regime}")
    if regime_result.worst_regime:
        lines.append(f"**Worst regime**: {regime_result.worst_regime}")
    lines.append("")

    # Regime insights
    if regime_result.by_trend:
        wr_values = {k: v.win_rate for k, v in regime_result.by_trend.items() if v.n_trades >= 2}
        if wr_values:
            best_wr = max(wr_values.values())
            worst_wr = min(wr_values.values())
            if best_wr - worst_wr > 20:
                lines.append(f"- Strategy has strong regime dependency: "
                             f"win rate ranges from {worst_wr:.0f}% to {best_wr:.0f}% "
                             f"across trend regimes.")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    # Write file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path


def _write_regime_table(lines, title, regime_dict, label_map):
    """Write a regime stats table to lines."""
    lines.append(f"### {title}")
    lines.append("")
    if not regime_dict:
        lines.append("*No data.*")
        lines.append("")
        return
    lines.append("| Regime | Trades | Win Rate | Avg PnL | Total PnL |")
    lines.append("|--------|:------:|:--------:|:-------:|:---------:|")
    for key, label in label_map.items():
        stats = regime_dict.get(key)
        if stats and stats.n_trades > 0:
            lines.append(
                f"| {label} | {stats.n_trades} | {stats.win_rate:.1f}% | "
                f"{stats.avg_pnl_pct:+.2f}% | {stats.total_pnl_pct:+.2f}% |"
            )
    lines.append("")


def run_full_attribution(test_results, load_strategy_fn, test_sets, output_dir=None):
    """Orchestrator: run attribution for all strategies with positive test Sharpe.

    Args:
        test_results: list of dicts from validate_all(), each with
            'version', 'params', 'test_AG', 'test_EC', etc.
        load_strategy_fn: callable(version) -> strategy class
        test_sets: dict like {"AG": ("2025-01-01", "2026-03-01"), ...}
        output_dir: directory for reports (default: research_log/attribution/)
    """
    from attribution.signal import run_signal_attribution
    from attribution.regime import run_regime_attribution

    if output_dir is None:
        from pathlib import Path
        output_dir = str(Path(__file__).resolve().parents[1] / "research_log" / "attribution")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for r in test_results:
        ver = r['version']
        params = r.get('params', {})

        # Find first test symbol with positive Sharpe
        for symbol, (start, end) in test_sets.items():
            sharpe = r.get(f'test_{symbol}')
            if sharpe is not None and sharpe > 0:
                try:
                    strategy_cls = load_strategy_fn(ver)
                    print(f"  Attribution: {ver} on {symbol}...", end=" ", flush=True)

                    signal_result = run_signal_attribution(
                        strategy_cls, params, symbol, start, end,
                    )
                    regime_result = run_regime_attribution(
                        strategy_cls, params, symbol, start, end,
                    )

                    out_path = os.path.join(output_dir, f"{ver}_{symbol}.md")
                    generate_attribution_report(signal_result, regime_result, out_path)
                    print(f"done ({out_path})")
                    count += 1
                except Exception as e:
                    print(f"FAILED ({e})")
                break  # One symbol per strategy

    print(f"\nAttribution reports generated: {count}/{len(test_results)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py::TestReportGenerator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/simon/Desktop/QBase
git add attribution/report.py tests/test_attribution.py
git commit -m "[attribution] implement Markdown report generator + orchestrator"
```

---

### Task 5: Pipeline Integration + CLAUDE.md Update

**Files:**
- Modify: `strategies/strong_trend/validate_and_iterate.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Wire attribution into validate_and_iterate.py**

Add the attribution call after the final `write_report()` in `validate_and_iterate.py`. At the end of the `__main__` block, before the final summary, add:

```python
    # Attribution analysis
    print(f"\n{'=' * 70}")
    print("ATTRIBUTION ANALYSIS")
    print(f"{'=' * 70}")

    from attribution.report import run_full_attribution
    from strategies.strong_trend.optimizer import load_strategy_class
    run_full_attribution(
        test_results=test_results,
        load_strategy_fn=load_strategy_class,
        test_sets=TEST_SETS,
    )
```

This goes right after the `write_report(test_results)` call inside the final block (after the iteration loop completes or breaks).

- [ ] **Step 2: Update CLAUDE.md — add Step 4.5 to pipeline overview**

In the pipeline overview table near the top of CLAUDE.md, add Step 4.5 between Step 4 and Step 5:

```
第一步         第二步           第三步            第四步           第四五步          第五步
指标池  ──→  策略开发  ──→  单策略优化  ──→  测试集验证  ──→  归因分析  ──→  Portfolio 构建
(320+个)     (编写策略)     (训练集调参)     (只读验证)      (信号+行情)     (组合+赋权)
```

Add a new section after "第四步" with:

```markdown
## 第四五步：归因分析（Attribution Analysis）

### 核心原则

**每个进入 Portfolio 的策略必须通过归因分析。** 不是只看 Sharpe，还要理解 alpha 来源。

### 两层自动归因

**1. 信号归因（Signal Attribution）**

对每个指标做 ablation test：替换为中性值，重跑回测，看 Sharpe 下降多少。

- Contribution = Baseline Sharpe - Ablated Sharpe
- 贡献 > 60% → 核心依赖，需要评估风险
- 贡献 < 5% → 冗余指标，考虑移除以降低复杂度

策略可声明 `INDICATOR_CONFIG` 精确控制中性值，否则自动发现（用中位数）。

**2. 行情归因（Regime Attribution）**

将每笔交易标注当时的行情状态，计算各 regime 下的胜率和 PnL：

| 维度 | 指标 | 分档 |
|------|------|------|
| 趋势强度 | ADX(14) | Strong (>25) / Weak (15-25) / None (<15) |
| 波动率 | ATR 百分位 | High (>75th) / Normal / Low (<25th) |
| 量能活跃度 | Volume / SMA(20) | Active (>1.5x) / Normal / Quiet (<0.7x) |

### 运行方式

归因分析自动集成到 `validate_and_iterate.py`，在测试集验证完成后自动运行。

报告输出到 `research_log/attribution/{version}_{symbol}.md`。

也可单独运行：

\```python
from attribution.signal import run_signal_attribution
from attribution.regime import run_regime_attribution
from attribution.report import generate_attribution_report

signal = run_signal_attribution(strategy_cls, params, "AG", "2025-01-01", "2026-03-01")
regime = run_regime_attribution(strategy_cls, params, "AG", "2025-01-01", "2026-03-01")
generate_attribution_report(signal, regime, "research_log/attribution/v12_AG.md")
\```

### INDICATOR_CONFIG 声明（推荐）

\```python
class StrongTrendV12(TimeSeriesStrategy):
    INDICATOR_CONFIG = [
        {'name': 'Aroon Oscillator', 'array_attr': '_aroon_osc', 'neutral_value': 100, 'role': 'trend'},
        {'name': 'PPO Histogram',    'array_attr': '_ppo_hist',   'neutral_value': 1.0, 'role': 'momentum'},
        {'name': 'Volume Momentum',  'array_attr': '_vol_mom',    'neutral_value': 2.0, 'role': 'volume'},
    ]
\```

不声明则自动发现（用数组中位数作为中性值）。声明后归因更准确。
```

- [ ] **Step 3: Commit**

```bash
cd /Users/simon/Desktop/QBase
git add strategies/strong_trend/validate_and_iterate.py CLAUDE.md
git commit -m "[attribution] integrate into pipeline + document Step 4.5 in CLAUDE.md"
```

---

### Task 6: Integration Test on v12

**Files:**
- No new files — this is a run-and-verify task

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/simon/Desktop/QBase && python -m pytest tests/test_attribution.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run attribution on v12 as integration test**

```bash
cd /Users/simon/Desktop/QBase && python -c "
from strategies.strong_trend.optimizer import load_strategy_class
from attribution.signal import run_signal_attribution
from attribution.regime import run_regime_attribution
from attribution.report import generate_attribution_report

cls = load_strategy_class('v12')
params = {'aroon_period': 20, 'ppo_fast': 19, 'ppo_slow': 29, 'vol_mom_period': 25, 'atr_trail_mult': 4.814}

print('Running signal attribution...')
sig = run_signal_attribution(cls, params, 'AG', '2025-01-01', '2026-03-01')
print(f'  Baseline Sharpe: {sig.baseline_sharpe}')
print(f'  Dominant: {sig.dominant_indicator}')
for name, c in sig.contributions.items():
    print(f'  {name}: contribution={c[\"contribution\"]:+.3f} ({c[\"pct_contribution\"]:.1f}%)')

print()
print('Running regime attribution...')
reg = run_regime_attribution(cls, params, 'AG', '2025-01-01', '2026-03-01')
print(f'  Total trades: {reg.total_trades}')
print(f'  Best regime: {reg.best_regime}')
print(f'  Worst regime: {reg.worst_regime}')

print()
print('Generating report...')
path = generate_attribution_report(sig, reg, 'research_log/attribution/v12_AG.md')
print(f'  Report: {path}')
"
```

Expected: Report generated at `research_log/attribution/v12_AG.md` with signal and regime data.

- [ ] **Step 3: Review the generated report**

Read `research_log/attribution/v12_AG.md` and verify it contains:
- Signal Attribution table with 3 indicators
- Regime Attribution tables for trend/volatility/activity
- Cross analysis table
- Auto-generated insights

- [ ] **Step 4: Commit generated report + push**

```bash
cd /Users/simon/Desktop/QBase
git add research_log/attribution/v12_AG.md
git commit -m "[attribution] v12 AG attribution report — first prototype"
git push
```

---

## Summary

| Task | What | Est. Time |
|------|------|-----------|
| 1 | Backtest helper (full result) | 3 min |
| 2 | Signal ablation engine | 8 min |
| 3 | Regime attribution + trade pairing | 8 min |
| 4 | Report generator + orchestrator | 5 min |
| 5 | Pipeline integration + CLAUDE.md | 5 min |
| 6 | Integration test on v12 | 5 min |
| **Total** | | **~35 min** |
