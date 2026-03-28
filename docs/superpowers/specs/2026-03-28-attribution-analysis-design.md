# Attribution Analysis Module — Design Spec

## Goal

Add a Step 4.5 (Attribution Analysis) to QBase's 5-step pipeline, between test-set validation and portfolio construction. Two fully automated analyses:

1. **Signal Attribution** — ablation test per indicator, quantifying each indicator's contribution to strategy performance
2. **Regime Attribution** — tag every trade with market regime state, compute per-regime win rate / PnL distribution

Output: one Markdown report per strategy, saved to `research_log/attribution/`.

---

## Architecture

```
QBase/
├── attribution/
│   ├── __init__.py
│   ├── signal.py        # Ablation test engine
│   ├── regime.py        # Regime tagging + per-regime stats
│   └── report.py        # Markdown report generator
├── research_log/
│   └── attribution/     # Generated reports (one .md per strategy)
```

All three modules are pure functions / stateless classes. No side effects except `report.py` writing files.

---

## Module 1: Signal Attribution (`attribution/signal.py`)

### Core Idea

For a strategy with N indicator arrays, run N+1 backtests:
- **Baseline**: full strategy, all indicators active → Sharpe_full
- **Ablation i**: replace indicator i's array with a "neutral" value (always-pass), re-run → Sharpe_without_i
- **Contribution of indicator i** = Sharpe_full - Sharpe_without_i

A high contribution means removing that indicator significantly hurts performance — it's a real alpha source. Near-zero contribution means it's dead weight.

### Interface

```python
def run_signal_attribution(
    strategy_cls,
    params: dict,
    symbol: str,
    start: str,
    end: str,
    freq: str = "daily",
    indicator_config: list[dict] | None = None,
) -> SignalAttributionResult:
    """
    Run ablation test for each indicator in the strategy.

    Args:
        strategy_cls: Strategy class (e.g., StrongTrendV12)
        params: Optimized parameters dict
        symbol: Symbol to test on (e.g., "AG")
        start, end: Test period
        freq: Bar frequency
        indicator_config: Optional list of indicator metadata. If None,
            auto-discovers from strategy instance.
            Each dict: {
                'name': str,           # Human-readable name (e.g., "Aroon Oscillator")
                'array_attr': str,     # Attribute name on strategy (e.g., "_aroon_osc")
                'neutral_value': float, # Value that makes this indicator "always pass"
                'role': str,           # 'trend' | 'momentum' | 'volume' | 'volatility' | 'filter'
            }

    Returns:
        SignalAttributionResult with per-indicator contributions
    """
```

### Auto-Discovery (when `indicator_config` is None)

1. Instantiate strategy, set params, run `on_init_arrays` once
2. Scan strategy instance for numpy array attributes (prefix `_`, exclude `_atr` which is used for stops)
3. For each discovered array, use **median value** as neutral (conservative: doesn't bias toward always-buy or always-sell)
4. Log a warning suggesting manual `indicator_config` for more accurate results

### Ablation Mechanics

```python
# Pseudocode
baseline_result = run_backtest(strategy, symbol, start, end, freq)
baseline_sharpe = baseline_result['sharpe']

contributions = {}
for indicator in indicator_config:
    # 1. Create fresh strategy instance with params
    modified_strategy = create_with_params(strategy_cls, params)

    # 2. Run on_init_arrays to populate all indicator arrays
    # (via a helper that calls engine internals)

    # 3. Replace target indicator array with neutral value
    original_array = getattr(modified_strategy, indicator['array_attr'])
    neutral_array = np.full_like(original_array, indicator['neutral_value'])
    setattr(modified_strategy, indicator['array_attr'], neutral_array)

    # 4. Run backtest (on_init_arrays already done, just run on_bar loop)
    ablated_result = run_backtest_with_precomputed(modified_strategy, symbol, start, end, freq)

    contributions[indicator['name']] = {
        'baseline_sharpe': baseline_sharpe,
        'ablated_sharpe': ablated_result['sharpe'],
        'contribution': baseline_sharpe - ablated_result['sharpe'],
        'pct_contribution': (baseline_sharpe - ablated_result['sharpe']) / max(abs(baseline_sharpe), 0.01),
        'role': indicator['role'],
    }
```

### Backtest Helper

We need a way to run a backtest where `on_init_arrays` is NOT called again (because we've already modified the arrays). Two options:

**Option A (Recommended)**: Wrap `on_init_arrays` to be a no-op after our manual modification:
```python
# After manually setting arrays:
modified_strategy._arrays_precomputed = True
original_on_init_arrays = modified_strategy.on_init_arrays
modified_strategy.on_init_arrays = lambda context, bars: None  # Skip re-computation
```
Then run normal `engine.run()`. The engine calls `on_init_arrays` but it's a no-op, so our modified arrays survive.

**Option B**: Fork `run_single_backtest` from `optimizer_core.py` into a version that skips `on_init_arrays`. More invasive, less maintainable.

We go with **Option A**.

### Result Dataclass

```python
@dataclass
class SignalAttributionResult:
    strategy_version: str
    symbol: str
    period: str                          # "2025-01-01 ~ 2026-03-01"
    baseline_sharpe: float
    baseline_trades: int
    contributions: dict[str, dict]       # indicator_name -> {ablated_sharpe, contribution, pct, role}
    dominant_indicator: str              # Highest contribution
    redundant_indicators: list[str]     # Contribution < 5% of baseline
```

### Edge Cases

- **Strategy fails without indicator**: ablated Sharpe = -999 → contribution = baseline + 999 (cap at baseline Sharpe)
- **Negative baseline Sharpe**: still run ablation, contributions are relative
- **Single-indicator strategy**: skip ablation, report "single indicator, no ablation possible"
- **Stop-loss array (_atr)**: always exclude from ablation (removing stops is meaningless)

---

## Module 2: Regime Attribution (`attribution/regime.py`)

### Core Idea

For each trade in the strategy's trade log, tag it with the market regime at entry time. Then compute per-regime statistics: win rate, avg PnL, number of trades.

### Regime Dimensions

Three orthogonal dimensions, each using simple, well-understood indicators:

| Dimension | Indicator | Computation | Bins |
|-----------|-----------|-------------|------|
| **Trend Strength** | ADX(14) | From QBase `indicators/trend/adx.py` | Strong (>25), Weak (15-25), None (<15) |
| **Volatility Regime** | ATR percentile | ATR(14) rank in trailing 252 bars | High (>75th), Normal (25-75th), Low (<25th) |
| **Volume Regime** | Volume vs 20MA | Current volume / SMA(volume, 20) | Active (>1.5), Normal (0.7-1.5), Quiet (<0.7) |

These three are always computed regardless of the strategy's own indicators — they're independent diagnostics.

### Interface

```python
def run_regime_attribution(
    strategy_cls,
    params: dict,
    symbol: str,
    start: str,
    end: str,
    freq: str = "daily",
) -> RegimeAttributionResult:
    """
    Run strategy, tag each trade with market regime, compute per-regime stats.

    Returns:
        RegimeAttributionResult with per-regime breakdowns
    """
```

### Implementation

```python
# Pseudocode
# 1. Run backtest, get result with trades DataFrame + equity curve
result = run_backtest_full(strategy, symbol, start, end, freq)
trades_df = result.trades  # columns: datetime, side, lots, price, ...

# 2. Load bar data, compute regime indicators on FULL array
bars = load_bars(symbol, freq, start, end)
closes = bars._close
highs = bars._high
lows = bars._low
volumes = bars._volume

adx_arr = adx(highs, lows, closes, period=14)
atr_arr = atr(highs, lows, closes, period=14)
vol_ma = sma(volumes, 20)

# 3. Compute regime labels for every bar
trend_regime = np.where(adx_arr > 25, 'strong', np.where(adx_arr > 15, 'weak', 'none'))
atr_pctile = rolling_percentile(atr_arr, 252)
vol_regime = np.where(atr_pctile > 0.75, 'high', np.where(atr_pctile > 0.25, 'normal', 'low'))
volume_ratio = volumes / vol_ma
activity_regime = np.where(volume_ratio > 1.5, 'active', np.where(volume_ratio > 0.7, 'normal', 'quiet'))

# 4. Match trades to regime at entry bar
# Need to map trade datetime → bar index
# Then tag each trade with regime state at that bar

# 5. Group trades by regime, compute stats per group
```

### Trade Pairing

The `trades` DataFrame from AlphaForge has individual orders (buy/sell), not paired round-trips. We need to pair entries and exits:

```python
def pair_trades(trades_df) -> list[dict]:
    """
    Pair entry and exit trades into round-trip records.

    Returns list of:
        {
            'entry_datetime': ...,
            'exit_datetime': ...,
            'entry_bar_idx': ...,   # Index in the bar array
            'exit_bar_idx': ...,
            'side': 1 or -1,       # 1=long, -1=short
            'entry_price': ...,
            'exit_price': ...,
            'lots': ...,
            'pnl': ...,            # Realized PnL
            'pnl_pct': ...,        # PnL as % of entry capital
            'holding_bars': ...,   # Exit bar - entry bar
        }
    """
```

Pairing logic: walk through trades chronologically, match opens with closes. Handle partial closes by splitting into sub-trades.

### Result Dataclass

```python
@dataclass
class RegimeAttributionResult:
    strategy_version: str
    symbol: str
    period: str
    total_trades: int
    total_sharpe: float

    # Per-regime breakdowns
    by_trend: dict[str, RegimeStats]     # 'strong' / 'weak' / 'none'
    by_volatility: dict[str, RegimeStats] # 'high' / 'normal' / 'low'
    by_activity: dict[str, RegimeStats]   # 'active' / 'normal' / 'quiet'

    # Cross-tabulation: trend x volatility (most informative combo)
    cross_trend_vol: dict[tuple, RegimeStats]

    # Best and worst regimes
    best_regime: str    # e.g., "strong trend + high volatility"
    worst_regime: str   # e.g., "no trend + low volatility"

@dataclass
class RegimeStats:
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    avg_holding_bars: float
    best_trade_pnl: float
    worst_trade_pnl: float
```

### Edge Cases

- **No trades in a regime**: report `n_trades=0`, skip stats
- **Regime indicator NaN at trade entry**: tag as `'unknown'`, report separately
- **Multi-bar trades**: use entry bar's regime (the decision point)

---

## Module 3: Report Generator (`attribution/report.py`)

### Interface

```python
def generate_attribution_report(
    signal_result: SignalAttributionResult,
    regime_result: RegimeAttributionResult,
    output_path: str,
) -> str:
    """Generate Markdown attribution report. Returns the file path."""
```

### Report Template

```markdown
# {version} Attribution Report — {symbol} ({period})

## Signal Attribution (Ablation Test)

| Indicator | Role | Baseline Sharpe | Without It | Contribution | % of Total |
|-----------|------|:-:|:-:|:-:|:-:|
| {name} | {role} | {baseline} | {ablated} | {contrib} | {pct}% |
| ... | ... | ... | ... | ... | ... |

**Dominant indicator**: {dominant} ({pct}% contribution)
**Redundant indicators**: {list or "None"}

### Interpretation
- {auto-generated insight based on contributions}

## Regime Attribution

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | {n} | {wr}% | {avg}% | {total}% |
| Weak (15-25) | ... | ... | ... | ... |
| None (<15) | ... | ... | ... | ... |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| High (>75th) | ... | ... | ... | ... |
| Normal (25-75th) | ... | ... | ... | ... |
| Low (<25th) | ... | ... | ... | ... |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | ... | ... | ... | ... |
| Normal | ... | ... | ... | ... |
| Quiet (<0.7x) | ... | ... | ... | ... |

### Cross Analysis: Trend x Volatility

| | High Vol | Normal Vol | Low Vol |
|--|:--:|:--:|:--:|
| **Strong Trend** | {wr}% ({n}) | ... | ... |
| **Weak Trend** | ... | ... | ... |
| **No Trend** | ... | ... | ... |

**Best regime**: {best} — {n} trades, {wr}% win rate, {avg_pnl}% avg PnL
**Worst regime**: {worst} — {n} trades, {wr}% win rate, {avg_pnl}% avg PnL

---

*Generated: {timestamp}*
```

### Auto-Generated Insights

The report module generates simple rule-based insights:

- If one indicator contributes > 60% → "Strategy is heavily dependent on {name}. Consider if this is a feature or a risk."
- If an indicator contributes < 5% → "{name} adds minimal value. Consider removing for simplicity."
- If win rate differs > 20pp between best and worst regime → "Strategy has strong regime dependency. Performs {X}% better in {regime}."
- If > 50% of trades are in one regime → "Strategy is concentrated in {regime}. {N}% of all trades."

---

## Pipeline Integration

### Step 4.5 in the Pipeline

```
Step 4: Test Validation → Step 4.5: Attribution → Step 5: Portfolio
```

Attribution runs after test-set validation, using the SAME test-set results. No additional optimization or parameter changes.

### Integration Point: `validate_and_iterate.py`

After `validate_all()` completes and `write_report()` is called, add attribution:

```python
# In validate_and_iterate.py main pipeline:
from attribution.report import run_full_attribution

# After final validation:
test_results = validate_all()
write_report(test_results)

# NEW: Attribution analysis on all strategies with positive test Sharpe
run_full_attribution(
    test_results=test_results,
    strategy_classes=STRATEGY_CLASSES,
    test_sets=TEST_SETS,
)
```

### `run_full_attribution` Orchestrator

```python
def run_full_attribution(test_results, strategy_classes, test_sets):
    """Run attribution for all strategies that passed validation."""
    for r in test_results:
        ver = r['version']
        params = r['params']

        # Run on first test symbol with positive Sharpe
        for symbol, (start, end) in test_sets.items():
            sharpe = r.get(f'test_{symbol}')
            if sharpe is not None and sharpe > 0:
                signal_result = run_signal_attribution(
                    strategy_classes[ver], params, symbol, start, end)
                regime_result = run_regime_attribution(
                    strategy_classes[ver], params, symbol, start, end)
                generate_attribution_report(
                    signal_result, regime_result,
                    f'research_log/attribution/{ver}_{symbol}.md')
                break  # One symbol per strategy is enough for attribution
```

### Indicator Config per Strategy

Each strategy can optionally define an `INDICATOR_CONFIG` class attribute:

```python
class StrongTrendV12(TimeSeriesStrategy):
    INDICATOR_CONFIG = [
        {'name': 'Aroon Oscillator', 'array_attr': '_aroon_osc', 'neutral_value': 100, 'role': 'trend'},
        {'name': 'PPO Histogram',    'array_attr': '_ppo_hist',   'neutral_value': 1.0, 'role': 'momentum'},
        {'name': 'Volume Momentum',  'array_attr': '_vol_mom',    'neutral_value': 2.0, 'role': 'volume'},
    ]
```

If not defined, the module auto-discovers arrays and uses median values. For the initial rollout, we add `INDICATOR_CONFIG` to the top strategies that will be analyzed first.

---

## Dependencies

- Uses existing `optimizer_core.run_single_backtest()` for running backtests
- Uses existing QBase indicators (`indicators/trend/adx.py`, `indicators/volatility/atr.py`)
- Uses AlphaForge `MarketDataLoader` for bar data
- No new external dependencies

## Performance

- Signal attribution: N+1 backtests per strategy (N = number of indicators, typically 3)
  - Daily strategy: ~0.4s × 4 = ~1.6s per strategy
  - 1h strategy: ~2.8s × 4 = ~11s per strategy
- Regime attribution: 1 backtest + indicator computation, negligible extra cost
- Total for 50 strategies: ~2-5 minutes (daily freq)

## Testing

- Unit test `pair_trades` with known trade sequences
- Unit test regime tagging with synthetic data
- Integration test: run on v12 AG, verify report is generated and contains expected sections
- Verify ablation produces different Sharpe when indicator is actually important

---

*Spec version: 1.0 | Date: 2026-03-28*
