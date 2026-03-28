"""
strategies/optimizer_core.py — Shared optimization infrastructure for QBase.

Provides:
1. Parameter auto-discovery from strategy class annotations
2. Composite objective function (Sharpe + consistency + drawdown penalty)
3. Two-phase optimization (coarse → fine)
4. Parameter robustness checking (plateau vs spike detection)
5. Multi-seed validation
6. Unified backtest runner with freq mapping and bar resampling
7. Unified result schema (build_result_entry, detect_strategy_status)

Used by:
- strategies/strong_trend/optimizer.py
- strategies/all_time/ag/optimizer.py
- strategies/all_time/i/optimizer.py
- strategies/boss/optimizer.py
- strategies/medium_trend/optimizer.py
"""
import sys
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings("ignore")


# =====================================================================
# 1. Parameter Auto-Discovery
# =====================================================================

SKIP_ATTRS = frozenset({
    "name", "warmup", "freq", "contract_multiplier",
})

KNOWN_RANGES = {
    "atr_trail_mult": (2.0, 5.5),
    "atr_stop_mult": (2.0, 5.5),
    "st_mult": (1.5, 5.0),
    "kc_mult": (1.0, 3.0),
    "chand_mult": (2.0, 5.0),
    "psar_af_step": (0.01, 0.04),
    "psar_af_max": (0.1, 0.3),
    "t3_vfactor": (0.5, 0.9),
    "alma_offset": (0.7, 0.95),
}


def auto_discover_params(strategy_cls):
    """Auto-discover tunable parameters from a strategy class.

    Discovery priority:
    1. AlphaForge @param decorator (get_tunable_params)
    2. AlphaForge annotation-based (get_annotation_params)
    3. Manual fallback via __annotations__ + defaults

    Returns:
        list of dicts: [{'name', 'low', 'high', 'step', 'dtype'}, ...]
    """
    for method_name in ("get_tunable_params", "get_annotation_params"):
        try:
            discovered = getattr(strategy_cls, method_name)()
            if discovered:
                return [
                    {
                        "name": p.name,
                        "low": p.low,
                        "high": p.high,
                        "step": p.step,
                        "dtype": _infer_dtype(strategy_cls, p.name),
                    }
                    for p in discovered
                ]
        except (AttributeError, TypeError, Exception):
            pass

    # Manual fallback: walk MRO to collect all annotations
    annotations = {}
    for cls in reversed(strategy_cls.__mro__):
        annotations.update(getattr(cls, "__annotations__", {}))

    params = []
    for attr_name, type_hint in annotations.items():
        if attr_name.startswith("_") or attr_name in SKIP_ATTRS:
            continue
        default = getattr(strategy_cls, attr_name, None)
        if default is None:
            continue

        is_int = type_hint is int or (
            isinstance(default, int) and not isinstance(default, bool)
        )
        is_float = type_hint is float or isinstance(default, float)
        if not (is_int or is_float):
            continue

        dtype = "int" if is_int else "float"
        low, high = _compute_range(attr_name, default, dtype)
        step = _compute_step(low, high, dtype)
        params.append(
            {"name": attr_name, "low": low, "high": high, "step": step, "dtype": dtype}
        )

    return params


def _infer_dtype(cls, name):
    default = getattr(cls, name, None)
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    return "float"


def _compute_range(name, default, dtype):
    """Compute parameter search range based on name patterns and known ranges."""
    if name in KNOWN_RANGES:
        return KNOWN_RANGES[name]

    period_kw = (
        "period", "lookback", "fast", "slow", "long", "short",
        "signal", "tenkan", "kijun", "wma", "ema", "base", "lag",
        "sum", "n_states", "hurst_lag",
    )
    if any(kw in name.lower() for kw in period_kw):
        if dtype == "int":
            return max(2, int(default * 0.4)), int(default * 3.0)
        return max(0.5, default * 0.4), default * 3.0

    threshold_kw = ("threshold", "thresh", "mult", "entry")
    if any(kw in name.lower() for kw in threshold_kw):
        return max(0.01, default * 0.3), default * 3.0

    # Generic
    if dtype == "int":
        return max(1, int(default * 0.4)), int(default * 3.0)
    return max(0.01, default * 0.3), default * 3.0


def _compute_step(low, high, dtype):
    if dtype == "int":
        return max(1, int((high - low) / 15))
    return round((high - low) / 15, 4)


def suggest_params(trial, param_specs):
    """Use Optuna trial to suggest parameters from spec list."""
    params = {}
    for spec in param_specs:
        name, low, high = spec["name"], spec["low"], spec["high"]
        if spec["dtype"] == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))
    return params


def create_strategy_with_params(strategy_cls, params):
    """Create a strategy instance and set parameter values."""
    instance = strategy_cls()
    for k, v in params.items():
        if hasattr(instance, k):
            setattr(instance, k, v)
    return instance


def narrow_param_space(param_specs, best_params, shrink_ratio=0.3):
    """Narrow search space around best params for fine-tuning phase.

    Each parameter's range is shrunk to ±(shrink_ratio/2) of original range
    centered on the best value, clamped to original bounds.

    Boundary protection: when best value is within 20% of a parameter boundary,
    uses a wider range (40% of original) anchored at that boundary to avoid
    getting trapped in a dead zone at the edge.
    """
    narrowed = []
    for spec in param_specs:
        name = spec["name"]
        orig_low, orig_high = spec["low"], spec["high"]
        orig_range = orig_high - orig_low

        if name in best_params:
            val = best_params[name]
            half_range = orig_range * shrink_ratio / 2
            new_low = max(orig_low, val - half_range)
            new_high = min(orig_high, val + half_range)

            # Boundary protection: if best is within 20% of boundary,
            # use a wider range centered on the boundary to avoid dead zone
            if (val - orig_low) < orig_range * 0.2:  # near low boundary
                new_low = orig_low
                new_high = min(orig_high, orig_low + orig_range * 0.4)
            elif (orig_high - val) < orig_range * 0.2:  # near high boundary
                new_high = orig_high
                new_low = max(orig_low, orig_high - orig_range * 0.4)

            new_step = spec["step"] / 2 if spec["step"] else spec["step"]
            narrowed.append({**spec, "low": new_low, "high": new_high, "step": new_step})
        else:
            narrowed.append(spec)
    return narrowed


# =====================================================================
# 2. Composite Objective Function
# =====================================================================

# Minimum trade counts by frequency.
# Lowered to accommodate short evaluation periods (2-4 month segments in medium trend).
# Old thresholds (30/50/80/100/150) killed all daily/4h strategies on short segments.
MIN_TRADES_BY_FREQ = {
    "daily": 10,
    "4h": 20,
    "1h": 30,      # V6: natively supported
    "60min": 30,    # Legacy alias for 1h
    "30min": 50,
    "15min": 80,
    "10min": 80,
    "5min": 80,
}


# ── Dimension Scoring Functions (each returns 0-10) ──

def _score_sharpe(sharpe, mode="tanh"):
    """Score Sharpe ratio on a 0-10 scale.

    Two modes:
    - 'tanh' (default, coarse phase): diminishing returns curve.
      Prevents optimizer from chasing extreme Sharpe at the cost of robustness.
        Sharpe 0.0 → 0.0     Sharpe 0.5 → 3.4
        Sharpe 1.0 → 6.0     Sharpe 1.5 → 7.8
        Sharpe 2.0 → 8.9     Sharpe 3.0 → 9.7

    - 'linear' (fine phase): linear scaling, capped at 10.
      Rewards absolute Sharpe improvements equally, pushing optimizer to
      maximize Sharpe within the already-validated parameter neighborhood.
        Sharpe 0.0 → 0.0     Sharpe 0.5 → 1.7
        Sharpe 1.0 → 3.3     Sharpe 1.5 → 5.0
        Sharpe 2.0 → 6.7     Sharpe 3.0 → 10.0

    Returns: float 0-10
    """
    if sharpe <= 0:
        return 0.0
    if mode == "linear":
        return float(min(10.0, sharpe * 10.0 / 3.0))
    return float(10.0 * np.tanh(0.7 * sharpe))


def _score_risk(max_dd):
    """Score risk control based on MaxDD. Lower drawdown = higher score.

    Piecewise linear:
        |dd| ≤  5% → 10      |dd| = 10% → 8
        |dd| = 15% → 6       |dd| = 20% → 4
        |dd| = 30% → 2       |dd| ≥ 40% → 0

    Returns: float 0-10
    """
    dd = abs(max_dd) if max_dd is not None else 0.0
    if dd <= 0.05:
        return 10.0
    if dd >= 0.40:
        return 0.0
    # Linear interpolation: 10 at 5%, 0 at 40%
    return float(max(0.0, 10.0 - (dd - 0.05) / 0.35 * 10.0))


def _score_quality(concentration):
    """Score profit quality based on concentration. Lower = better.

    Piecewise linear:
        conc ≤ 0.3 → 10      conc = 0.5 → 7
        conc = 0.7 → 4       conc ≥ 0.95 → 0

    Returns: float 0-10
    """
    if concentration is None:
        return 5.0  # neutral when no data
    c = max(0.0, min(1.0, concentration))
    if c <= 0.30:
        return 10.0
    if c >= 0.95:
        return 0.0
    # Linear: 10 at 0.3, 0 at 0.95
    return float(max(0.0, 10.0 - (c - 0.30) / 0.65 * 10.0))


def _score_stability(monthly_win_rate):
    """Score stability based on monthly win rate.

    Piecewise linear:
        wr ≥ 65% → 10     wr = 55% → 7
        wr = 45% → 4      wr ≤ 30% → 0

    Returns: float 0-10
    """
    if monthly_win_rate is None:
        return 5.0  # neutral when no data
    wr = max(0.0, min(1.0, monthly_win_rate))
    if wr >= 0.65:
        return 10.0
    if wr <= 0.30:
        return 0.0
    # Linear: 0 at 30%, 10 at 65%
    return float(max(0.0, (wr - 0.30) / 0.35 * 10.0))


def _compute_monthly_win_rate(equity_curve):
    """Compute monthly win rate from equity curve.

    A month is a 'win' if the equity at month-end > equity at month-start.

    Returns: float 0-1 (fraction of winning months), or None if insufficient data.
    """
    try:
        eq = np.asarray(equity_curve, dtype=np.float64)
        if hasattr(equity_curve, "values"):
            eq = equity_curve.values.astype(np.float64)

        if len(eq) < 30:
            return None

        # Sample roughly monthly (every 20-22 trading days)
        step = 21
        monthly_values = eq[::step]
        if len(monthly_values) < 3:
            return None

        monthly_returns = np.diff(monthly_values) / np.maximum(np.abs(monthly_values[:-1]), 1e-9)
        monthly_returns = monthly_returns[np.isfinite(monthly_returns)]
        if len(monthly_returns) < 3:
            return None

        win_rate = float(np.sum(monthly_returns > 0) / len(monthly_returns))
        return win_rate

    except Exception:
        return None


# ── Dimension Weights ──

W_SHARPE = 0.60
W_RISK = 0.15
W_QUALITY = 0.15
W_STABILITY = 0.10


def composite_objective(results, min_valid=1, freq=None, scoring_mode="tanh"):
    """Weighted multi-dimensional objective function (0-10 scale).

    Scores each dimension 0-10, then computes weighted sum:

        score = 0.60 × S_sharpe + 0.15 × S_risk + 0.15 × S_quality + 0.10 × S_stability

    Dimensions:
    - **S_sharpe (60%)**: Sharpe scoring (mode depends on optimization phase).
      'tanh' (coarse): diminishing returns, prevents chasing extreme Sharpe.
      'linear' (fine): equal reward for each Sharpe increment, maximizes absolute Sharpe.
    - **S_risk (15%)**: MaxDD scoring — |dd|≤5% → 10, |dd|=20% → 4, |dd|≥40% → 0.
    - **S_quality (15%)**: profit concentration — conc≤0.3 → 10, conc=0.7 → 4, conc≥0.95 → 0.
    - **S_stability (10%)**: monthly win rate — wr≥65% → 10, wr=45% → 4, wr≤30% → 0.
      Falls back to 5 (neutral) when equity curve is unavailable.

    For multi-symbol evaluation, S_sharpe uses the mean Sharpe scaled by
    a consistency factor (penalizes when any symbol has negative Sharpe).

    Trade count filter: results below frequency-based minimum are excluded.

    Args:
        results: list of dicts with keys 'sharpe', and optionally
                 'max_drawdown', 'n_trades', 'profit_concentration', 'monthly_win_rate'.
        min_valid: minimum valid results after filtering.
        freq: strategy frequency for trade count threshold.
        scoring_mode: 'tanh' (coarse phase, default) or 'linear' (fine phase).

    Returns:
        float: weighted score (0-10 range), or -10.0 if insufficient valid results.
    """
    min_trades = MIN_TRADES_BY_FREQ.get(freq, 0) if freq else 0

    valid = []
    for r in results:
        if r.get("sharpe", -999) <= -900:
            continue
        if min_trades > 0 and r.get("n_trades") is not None:
            if r["n_trades"] < min_trades:
                continue
        valid.append(r)

    if len(valid) < min_valid:
        return -10.0

    # ── S_sharpe (60%) ──
    sharpes = [r["sharpe"] for r in valid]
    mean_sharpe = float(np.mean(sharpes))

    if len(valid) > 1:
        # Multi-symbol: penalize if any symbol has negative Sharpe
        min_sharpe = float(np.min(sharpes))
        if min_sharpe < 0:
            # Scale down: if worst symbol is very negative, reduce effective Sharpe
            consistency = max(0.5, 1.0 + 0.3 * (min_sharpe / max(abs(mean_sharpe), 0.01)))
            effective_sharpe = mean_sharpe * min(1.0, consistency)
        else:
            effective_sharpe = mean_sharpe
    else:
        effective_sharpe = mean_sharpe

    s_sharpe = _score_sharpe(effective_sharpe, mode=scoring_mode)

    # ── S_risk (15%) ──
    drawdowns = [r["max_drawdown"] for r in valid if r.get("max_drawdown") is not None]
    if drawdowns:
        mean_abs_dd = float(np.mean([abs(d) for d in drawdowns]))
        s_risk = _score_risk(mean_abs_dd)
    else:
        s_risk = 5.0  # neutral

    # ── S_quality (15%) ──
    concentrations = [
        r["profit_concentration"]
        for r in valid
        if r.get("profit_concentration") is not None
    ]
    if concentrations:
        s_quality = _score_quality(float(np.mean(concentrations)))
    else:
        s_quality = 5.0  # neutral

    # ── S_stability (10%) ──
    win_rates = [
        r["monthly_win_rate"]
        for r in valid
        if r.get("monthly_win_rate") is not None
    ]
    if win_rates:
        s_stability = _score_stability(float(np.mean(win_rates)))
    else:
        s_stability = 5.0  # neutral

    return float(
        W_SHARPE * s_sharpe
        + W_RISK * s_risk
        + W_QUALITY * s_quality
        + W_STABILITY * s_stability
    )


# =====================================================================
# 3. Backtest Runner
# =====================================================================

def map_freq(freq):
    """Map strategy frequency to (AlphaForge load freq, resample factor).

    AlphaForge V6 native: 1min, 5min, 10min, 15min, 30min, 60min, 1h, daily.
    Note: V6 supports "1h" natively — no need to map to "60min".
    Non-native frequencies are loaded at a base freq and resampled.

    Returns:
        (load_freq, resample_factor) where factor=1 means no resample.
    """
    # V6: "1h" is now natively supported alongside "60min"
    NATIVE = {"1min", "5min", "10min", "15min", "30min", "60min", "1h", "daily"}
    if freq in NATIVE:
        return freq, 1

    RESAMPLE_MAP = {
        "20min": ("10min", 2),
        "4h": ("60min", 4),
    }
    if freq in RESAMPLE_MAP:
        return RESAMPLE_MAP[freq]

    return freq, 1


def resample_bars(bars, step):
    """Resample BarArray by grouping every ``step`` bars.

    E.g. step=4 converts 60min bars to 4h bars.

    .. deprecated:: V6
        AlphaForge V6 supports native multi-TF via ``resample_freqs`` and
        ``context.get_resampled_bars()``.  This manual resample is kept for
        backward compatibility with existing optimizer/attribution code that
        loads bars outside of a strategy context.
    """
    from alphaforge.data.bardata import BarArray

    n = len(bars)
    indices = list(range(step - 1, n, step))
    if not indices:
        return bars
    new_len = len(indices)

    def _agg(src, mode):
        arr = np.array(src, dtype=np.float64)
        out = np.empty(new_len, dtype=np.float64)
        for j, end_idx in enumerate(indices):
            s = max(0, end_idx - step + 1)
            chunk = arr[s : end_idx + 1]
            if mode == "first":
                out[j] = chunk[0]
            elif mode == "max":
                out[j] = chunk.max()
            elif mode == "min":
                out[j] = chunk.min()
            elif mode == "last":
                out[j] = chunk[-1]
            elif mode == "sum":
                out[j] = chunk.sum()
        return out

    def _pick(src):
        return np.array(src)[indices]

    return BarArray(
        datetime_arr=_pick(bars.datetime),
        open_arr=_agg(bars.open, "first"),
        high_arr=_agg(bars.high, "max"),
        low_arr=_agg(bars.low, "min"),
        close_arr=_agg(bars.close, "last"),
        volume_arr=_agg(bars.volume, "sum"),
        amount_arr=_agg(bars.amount, "sum"),
        oi_arr=_agg(bars.oi, "last"),
        trading_day_arr=_pick(bars.trading_day),
        open_raw_arr=_agg(bars.open_raw, "first"),
        high_raw_arr=_agg(bars.high_raw, "max"),
        low_raw_arr=_agg(bars.low_raw, "min"),
        close_raw_arr=_agg(bars.close_raw, "last"),
        origin_symbol_arr=_pick(bars.origin_symbol),
        factor_arr=_pick(bars.factor),
        is_rollover_arr=_pick(bars.is_rollover),
    )


def _compute_profit_concentration(equity_curve):
    """Compute profit concentration ratio from an equity curve.

    Measures what fraction of total gross profit comes from the top 10%
    of days. A strategy where 80%+ of profit comes from a handful of
    days is fragile — remove those days and there's nothing left.

    Method:
    1. Compute daily PnL from equity curve
    2. Sum all profitable days → total_profit
    3. Take the top 10% of ALL days by PnL
    4. concentration = sum(top_10%_pnl) / total_profit

    For a well-diversified strategy (normal returns, Sharpe~1):
        concentration ≈ 0.3 - 0.5 (healthy)
    For a spike-dependent strategy:
        concentration ≈ 0.8 - 1.0 (fragile)

    Returns:
        float in [0, 1]: 0 = evenly distributed, 1 = all profit from top 10%.
        None if equity curve is too short or invalid.
    """
    try:
        eq = np.asarray(equity_curve, dtype=np.float64)
        if hasattr(equity_curve, "values"):
            eq = equity_curve.values.astype(np.float64)

        if len(eq) < 30:
            return None

        # Daily PnL (absolute, not percentage — avoids compounding distortion)
        daily_pnl = np.diff(eq)
        daily_pnl = daily_pnl[np.isfinite(daily_pnl)]
        if len(daily_pnl) < 30:
            return None

        # Total gross profit (sum of all profitable days)
        total_profit = float(np.sum(daily_pnl[daily_pnl > 0]))
        if total_profit <= 0:
            return None  # no profitable days, concentration not meaningful

        # Top 10% of ALL days by PnL (descending)
        n_top = max(1, int(len(daily_pnl) * 0.10))
        sorted_desc = np.sort(daily_pnl)[::-1]
        top_10_pnl = float(np.sum(sorted_desc[:n_top]))

        # Clamp to [0, 1]
        concentration = max(0.0, min(1.0, top_10_pnl / total_profit))
        return concentration

    except Exception:
        return None


def _build_backtest_config(config_mode, initial_capital, slippage_ticks, backtest_config):
    """Build a BacktestConfig for V6, falling back to legacy kwargs.

    Args:
        config_mode: "basic" (legacy positional args), "industrial" (V6 recommended),
                     or "custom" (caller provides a BacktestConfig instance).
        initial_capital: Used for "basic" and "industrial" modes.
        slippage_ticks: Used for "basic" and "industrial" modes.
        backtest_config: A pre-built BacktestConfig instance (only used when
                         config_mode="custom").

    Returns:
        A BacktestConfig instance, or None if V6 BacktestConfig is not available
        (graceful fallback to legacy constructor).
    """
    try:
        from alphaforge.engine.config import BacktestConfig
    except ImportError:
        # AlphaForge version < 6 — BacktestConfig not available
        return None

    if config_mode == "custom" and backtest_config is not None:
        return backtest_config

    if config_mode == "industrial":
        return BacktestConfig(
            initial_capital=initial_capital,
            slippage_ticks=slippage_ticks,
            volume_adaptive_spread=True,
            dynamic_margin=True,
            time_varying_spread=True,
            rollover_window_bars=20,
            margin_check_mode="daily",
            margin_call_grace_bars=3,
            asymmetric_impact=True,
            detect_locked_limit=True,
        )

    # "basic" mode — minimal config, mirrors legacy behaviour
    return BacktestConfig(
        initial_capital=initial_capital,
        slippage_ticks=slippage_ticks,
    )


def run_single_backtest(
    strategy,
    symbol,
    start,
    end,
    freq="daily",
    data_dir=None,
    initial_capital=1_000_000,
    slippage_ticks=1.0,
    config_mode="basic",
    backtest_config=None,
):
    """Run a single backtest and return a result dict.

    Args:
        strategy: Strategy instance.
        symbol: Trading symbol string.
        start: Start date string.
        end: End date string.
        freq: Frequency string ("daily", "1h", "4h", etc.).
        data_dir: Data directory override.
        initial_capital: Initial capital (used when config_mode != "custom").
        slippage_ticks: Slippage in ticks (used when config_mode != "custom").
        config_mode: "basic" (default, legacy), "industrial" (V6 recommended
                     settings for realistic simulation), or "custom" (use the
                     provided ``backtest_config`` instance directly).
        backtest_config: A pre-built ``BacktestConfig`` instance.  Only used
                         when ``config_mode="custom"``.

    Returns:
        dict with keys:
            sharpe (float): Sharpe ratio, or -999.0 on failure
            max_drawdown (float|None): Maximum drawdown (negative)
            n_trades (int|None): Total trade count
            total_return (float|None): Total return
            profit_concentration (float|None): 0=even, 1=concentrated in top 10% days
            monthly_win_rate (float|None): Monthly win rate
    """
    FAIL = {
        "sharpe": -999.0,
        "max_drawdown": None,
        "n_trades": None,
        "total_return": None,
        "profit_concentration": None,
        "monthly_win_rate": None,
    }
    try:
        from alphaforge.data.market import MarketDataLoader
        from alphaforge.data.contract_specs import ContractSpecManager
        from alphaforge.engine.event_driven import EventDrivenBacktester

        if data_dir is None:
            from config import get_data_dir
            data_dir = get_data_dir()

        loader = MarketDataLoader(data_dir)
        load_freq, resample_factor = map_freq(freq)
        bars = loader.load(symbol, freq=load_freq, start=start, end=end)
        if bars is None or len(bars) < strategy.warmup + 20:
            return FAIL

        if resample_factor > 1:
            bars = resample_bars(bars, resample_factor)
        if bars is None or len(bars) < strategy.warmup + 20:
            return FAIL

        # V6: Try to use BacktestConfig if available
        bt_config = _build_backtest_config(
            config_mode, initial_capital, slippage_ticks, backtest_config,
        )
        if bt_config is not None:
            engine = EventDrivenBacktester(
                spec_manager=ContractSpecManager(),
                config=bt_config,
            )
        else:
            # Legacy fallback (AlphaForge < V6)
            engine = EventDrivenBacktester(
                spec_manager=ContractSpecManager(),
                initial_capital=initial_capital,
                slippage_ticks=slippage_ticks,
            )
        result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)

        sharpe = float(result.sharpe)
        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = -999.0

        max_dd = getattr(result, "max_drawdown", None)
        if max_dd is not None and (np.isnan(max_dd) or np.isinf(max_dd)):
            max_dd = None

        n_trades = getattr(result, "n_trades", None) or getattr(
            result, "total_trades", None
        )
        total_return = getattr(result, "total_return", None)

        # Extract equity curve for quality/stability metrics
        profit_conc = None
        monthly_wr = None
        equity_curve = getattr(result, "equity_curve", None)
        if equity_curve is None:
            equity_curve = getattr(result, "equity", None)
        if equity_curve is not None:
            profit_conc = _compute_profit_concentration(equity_curve)
            monthly_wr = _compute_monthly_win_rate(equity_curve)

        return {
            "sharpe": sharpe,
            "max_drawdown": float(max_dd) if max_dd is not None else None,
            "n_trades": int(n_trades) if n_trades is not None else None,
            "total_return": float(total_return) if total_return is not None else None,
            "profit_concentration": profit_conc,
            "monthly_win_rate": monthly_wr,
        }
    except Exception:
        return FAIL


# =====================================================================
# 4. Parameter Robustness Analysis
# =====================================================================

def check_robustness(
    evaluate_fn, best_params, param_specs, n_neighbors=None, min_plateau_ratio=0.6
):
    """Check if best params sit on a stable plateau or a noise spike.

    Perturbs each parameter by ±15% of its range and re-evaluates.
    If most neighbors also have good scores, the optimum is robust.

    The number of neighbors scales with parameter dimensionality:
    ``max(20, n_params * 5)`` by default, ensuring adequate coverage
    even in higher-dimensional spaces.

    Args:
        evaluate_fn: callable(params_dict) -> float score
        best_params: dict of best parameter values
        param_specs: list of param spec dicts
        n_neighbors: number of random neighbors to sample.
                     None = auto-scale by param count (max(20, n_params * 5)).
        min_plateau_ratio: fraction of neighbors that must score > 50% of best

    Returns:
        dict with: is_robust, best_value, neighbor_mean, neighbor_std,
                   above_threshold_pct, plateau_params, n_neighbors_used
    """
    if n_neighbors is None:
        n_neighbors = max(20, len(param_specs) * 5)

    best_value = evaluate_fn(best_params)
    rng = np.random.RandomState(42)

    neighbor_values = []
    neighbor_params_list = []

    for _ in range(n_neighbors):
        perturbed = {}
        for spec in param_specs:
            name = spec["name"]
            low, high = spec["low"], spec["high"]
            val = best_params.get(name, (low + high) / 2)
            delta = rng.uniform(-0.15, 0.15) * (high - low)
            new_val = float(np.clip(val + delta, low, high))
            if spec["dtype"] == "int":
                new_val = int(round(new_val))
            perturbed[name] = new_val

        score = evaluate_fn(perturbed)
        neighbor_values.append(score)
        neighbor_params_list.append(perturbed)

    neighbor_mean = float(np.mean(neighbor_values))
    neighbor_std = float(np.std(neighbor_values))

    # Threshold: neighbor must achieve > 50% of best value
    if best_value > 0:
        threshold = best_value * 0.5
    else:
        threshold = best_value * 1.5  # for negative best, 1.5x is less negative

    above_count = sum(1 for v in neighbor_values if v > threshold)
    above_pct = above_count / max(1, len(neighbor_values))
    is_robust = above_pct >= min_plateau_ratio

    # If robust, check if any neighbor is even better (plateau center)
    plateau_params = dict(best_params)
    if is_robust and neighbor_values:
        best_neighbor_idx = int(np.argmax(neighbor_values))
        if neighbor_values[best_neighbor_idx] > best_value:
            plateau_params = neighbor_params_list[best_neighbor_idx]

    return {
        "is_robust": is_robust,
        "best_value": best_value,
        "neighbor_mean": neighbor_mean,
        "neighbor_std": neighbor_std,
        "above_threshold_pct": above_pct,
        "plateau_params": plateau_params,
        "n_neighbors_used": n_neighbors,
    }


# =====================================================================
# 5. Two-Phase Optimization
# =====================================================================

def optimize_two_phase(
    objective_fn,
    param_specs,
    coarse_trials=30,
    fine_trials=50,
    seed=42,
    study_name=None,
    verbose=True,
    probe_trials=5,
):
    """Two-phase optimization: coarse search → fine-tune → robustness check.

    Phase 1 (coarse): broad search with ``coarse_trials``.
    Phase 2 (fine): narrow search (±15% of range) around coarse best.
    Phase 3: robustness check on the final best.

    Built-in probe: first ``probe_trials`` are checked for all-failure
    (-999 or worse), in which case optimization is aborted early.

    Args:
        objective_fn: callable(params_dict) -> float (higher is better)
        param_specs: list of param spec dicts from auto_discover_params
        coarse_trials: number of coarse-phase trials
        fine_trials: number of fine-phase trials
        seed: random seed for TPE sampler
        study_name: optional Optuna study name
        verbose: print progress
        probe_trials: number of initial trials to check for all-failure

    Returns:
        dict with: best_params, best_value, coarse_best, fine_best,
                   robustness, n_trials, phase, early_stopped
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Phase 1: Coarse search ──
    if verbose:
        print(f"  Phase 1: Coarse ({coarse_trials} trials, probe={probe_trials})...")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=f"{study_name}_coarse" if study_name else None,
    )

    def _objective(trial):
        params = suggest_params(trial, param_specs)
        return objective_fn(params, scoring_mode="tanh")

    # Probe: run a few trials first to detect broken strategies
    actual_probe = min(probe_trials, coarse_trials)
    study.optimize(_objective, n_trials=actual_probe, show_progress_bar=False)

    probe_values = [t.value for t in study.trials if t.value is not None]
    all_failed = all(v <= -900 for v in probe_values) if probe_values else True

    if all_failed:
        if verbose:
            print(f"  EARLY STOP: All {actual_probe} probe trials failed. Strategy broken.")
        return {
            "best_params": {},
            "best_value": -999.0,
            "coarse_best": -999.0,
            "fine_best": None,
            "robustness": None,
            "n_trials": actual_probe,
            "phase": "probe_failed",
            "early_stopped": True,
        }

    # Run remaining coarse trials
    remaining_coarse = coarse_trials - actual_probe
    if remaining_coarse > 0:
        study.optimize(_objective, n_trials=remaining_coarse, show_progress_bar=verbose)

    coarse_best_value = study.best_value
    coarse_best_params = dict(study.best_params)

    if verbose:
        print(f"  Coarse best: {coarse_best_value:.4f}")

    # Skip fine phase if coarse result is very bad
    if coarse_best_value <= -5.0:
        return {
            "best_params": coarse_best_params,
            "best_value": coarse_best_value,
            "coarse_best": coarse_best_value,
            "fine_best": None,
            "robustness": None,
            "n_trials": coarse_trials,
            "phase": "coarse_only",
            "early_stopped": False,
        }

    # ── Probe validation: verify coarse best params before fine phase ──
    coarse_probe_score = objective_fn(coarse_best_params, scoring_mode="tanh")
    if coarse_probe_score <= -5.0:
        if verbose:
            print(f"  Skipping fine phase: coarse best params invalid (score={coarse_probe_score:.4f})")
        return {
            "best_params": coarse_best_params,
            "best_value": coarse_best_value,
            "coarse_best": coarse_best_value,
            "fine_best": None,
            "robustness": None,
            "n_trials": coarse_trials,
            "phase": "coarse_only",
            "early_stopped": False,
        }

    # ── Phase 2: Fine-tune ──
    if verbose:
        print(f"  Phase 2: Fine-tune ({fine_trials} trials)...")

    fine_specs = narrow_param_space(param_specs, coarse_best_params)

    study_fine = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed + 1000),
        study_name=f"{study_name}_fine" if study_name else None,
    )

    def _fine_objective(trial):
        params = suggest_params(trial, fine_specs)
        return objective_fn(params, scoring_mode="linear")

    study_fine.optimize(_fine_objective, n_trials=fine_trials, show_progress_bar=verbose)

    # Pick the better result from coarse vs fine
    if study_fine.best_value >= coarse_best_value:
        best_params = dict(study_fine.best_params)
        best_value = study_fine.best_value
    else:
        best_params = coarse_best_params
        best_value = coarse_best_value

    if verbose:
        print(f"  Fine best: {study_fine.best_value:.4f} → overall: {best_value:.4f}")

    # ── Phase 3: Robustness check ──
    if verbose:
        print("  Phase 3: Robustness check...")

    robustness = check_robustness(objective_fn, best_params, param_specs)

    if verbose:
        tag = "PLATEAU" if robustness["is_robust"] else "SPIKE"
        print(
            f"  Robustness: {tag} — neighbor_mean={robustness['neighbor_mean']:.4f}, "
            f"std={robustness['neighbor_std']:.4f}, "
            f"above_threshold={robustness['above_threshold_pct']:.0%}"
        )

    # Use plateau-adjusted params if robust
    final_params = (
        robustness["plateau_params"] if robustness["is_robust"] else best_params
    )
    final_value = objective_fn(final_params)

    return {
        "best_params": final_params,
        "best_value": final_value,
        "coarse_best": coarse_best_value,
        "fine_best": study_fine.best_value,
        "robustness": {
            "is_robust": robustness["is_robust"],
            "neighbor_mean": robustness["neighbor_mean"],
            "neighbor_std": robustness["neighbor_std"],
            "above_threshold_pct": robustness["above_threshold_pct"],
        },
        "n_trials": coarse_trials + fine_trials,
        "phase": "two_phase",
        "early_stopped": False,
    }


# =====================================================================
# 6. Multi-Seed Optimization
# =====================================================================

def optimize_multi_seed(
    objective_fn,
    param_specs,
    coarse_trials=30,
    fine_trials=50,
    seeds=(42, 123, 456),
    verbose=True,
    study_name=None,
    probe_trials=5,
):
    """Multi-seed optimization: run two-phase with multiple seeds,
    select the median result for stability.

    Runs the full two-phase pipeline independently with each seed.
    The final result is the one with the **median** best_value,
    which is more stable than picking the maximum.

    Cross-seed consistency is checked: if std(best_values) > 50% of
    mean(best_values), the result is flagged as inconsistent.

    Args:
        objective_fn: callable(params_dict) -> float
        param_specs: list of param spec dicts
        coarse_trials: per-seed coarse trials
        fine_trials: per-seed fine trials
        seeds: tuple of random seeds
        verbose: print progress
        study_name: optional base study name
        probe_trials: probe trials per seed

    Returns:
        dict with: best_params, best_value, seed_results, selected_seed,
                   robustness, all_robust, cross_seed_std, is_consistent,
                   n_trials_total
    """
    seed_results = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Seed {seed} ({i + 1}/{len(seeds)})")
            print(f"{'─'*50}")

        result = optimize_two_phase(
            objective_fn,
            param_specs,
            coarse_trials=coarse_trials,
            fine_trials=fine_trials,
            seed=seed,
            study_name=f"{study_name}_s{seed}" if study_name else None,
            verbose=verbose,
            probe_trials=probe_trials,
        )
        result["seed"] = seed
        seed_results.append(result)

        # If first seed's probe failed, all seeds will likely fail too
        if i == 0 and result.get("early_stopped"):
            if verbose:
                print("  First seed failed probe — skipping remaining seeds.")
            return {
                "best_params": {},
                "best_value": -999.0,
                "seed_results": seed_results,
                "selected_seed": seed,
                "robustness": None,
                "all_robust": False,
                "cross_seed_std": 0.0,
                "is_consistent": False,
                "n_trials_total": result["n_trials"],
                "early_stopped": True,
            }

    # Select median result (most stable)
    values = [r["best_value"] for r in seed_results]
    sorted_indices = np.argsort(values)
    median_idx = int(sorted_indices[len(sorted_indices) // 2])
    selected = seed_results[median_idx]

    # Cross-seed consistency check
    value_std = float(np.std(values))
    value_mean = float(np.mean(values))
    is_consistent = (
        value_std < 0.5 * abs(value_mean) if abs(value_mean) > 0.01 else value_std < 0.3
    )

    if verbose:
        print(f"\n{'═'*50}")
        print("Multi-seed summary")
        print(f"{'═'*50}")
        for r in seed_results:
            robust_tag = (
                "R" if r.get("robustness", {}).get("is_robust") else "F"
            )
            print(f"  Seed {r['seed']}: {r['best_value']:.4f} [{robust_tag}]")
        print(
            f"  Selected: seed {selected['seed']} (median) = {selected['best_value']:.4f}"
        )
        tag = "CONSISTENT" if is_consistent else "INCONSISTENT"
        print(f"  Cross-seed std: {value_std:.4f} — {tag}")

    return {
        "best_params": selected["best_params"],
        "best_value": selected["best_value"],
        "seed_results": [
            {"seed": r["seed"], "best_value": r["best_value"], "phase": r["phase"]}
            for r in seed_results
        ],
        "selected_seed": selected["seed"],
        "robustness": selected.get("robustness"),
        "all_robust": all(
            r.get("robustness", {}).get("is_robust", False) for r in seed_results
        ),
        "cross_seed_std": value_std,
        "is_consistent": is_consistent,
        "n_trials_total": sum(r["n_trials"] for r in seed_results),
        "early_stopped": False,
    }


# =====================================================================
# 7. Unified Result Schema
# =====================================================================

# All required fields in a standardized optimization result entry.
UNIFIED_SCHEMA_FIELDS = {
    "version", "freq", "best_sharpe", "best_score", "best_params",
    "n_trials", "phase", "status", "robustness", "elapsed_seconds",
}


def build_result_entry(
    version: str,
    freq: str,
    best_params: dict,
    sharpe: float = None,
    score: float = None,
    n_trials: int = 0,
    phase: str = "two_phase",
    robustness: dict = None,
    elapsed: float = 0.0,
    status: str = "active",
    **extra_fields,
) -> dict:
    """Build a standardized optimization result entry.

    All optimizers should call this to produce consistent output format.
    Extra fields (e.g. early_stopped, multi_seed, cross_seed_std) are
    preserved as-is for backward compatibility.

    Args:
        version: Strategy version (e.g. "v1", "v12").
        freq: Strategy frequency ("daily", "4h", "1h", "5min").
        best_params: Optimized parameter dict.
        sharpe: Raw Sharpe ratio (best_sharpe).
        score: Composite score 0-10 (best_score).
        n_trials: Total optimization trials.
        phase: Optimization phase ("coarse_only", "two_phase", "multi_seed", "probe_failed").
        robustness: Robustness check result dict, or None.
        elapsed: Wall-clock seconds.
        status: "active", "dead", "error", "import_error".
        **extra_fields: Additional fields preserved for backward compatibility.

    Returns:
        dict with all unified schema fields plus any extras.
    """
    entry = {
        "version": version,
        "freq": freq,
        "best_sharpe": round(sharpe, 4) if sharpe is not None else None,
        "best_score": round(score, 4) if score is not None else None,
        "best_params": best_params,
        "n_trials": n_trials,
        "phase": phase,
        "status": status,
        "robustness": robustness,
        "elapsed_seconds": round(elapsed, 1),
    }
    # Preserve extra fields for backward compatibility
    entry.update(extra_fields)
    return entry


def detect_strategy_status(result: dict) -> str:
    """Detect if a strategy is dead/error based on optimization results.

    Returns:
        "error" — if sharpe or score indicates complete failure (-999 etc.)
        "import_error" — if the result contains an import error marker
        "dead" — if the strategy produced 0 trades or negligible results
        "active" — otherwise
    """
    # Check for explicit error markers
    if result.get("error"):
        error_msg = str(result["error"]).lower()
        if "import" in error_msg or "module" in error_msg:
            return "import_error"
        return "error"

    sharpe = result.get("best_sharpe")
    score = result.get("best_score")

    # Check for sentinel failure values
    if sharpe is not None and sharpe <= -900:
        return "error"
    if score is not None and score <= -900:
        return "error"

    # Check for empty params (probe failed / no trades)
    if not result.get("best_params"):
        return "dead"

    return "active"


def is_strategy_dead(results_file: str, version: str) -> bool:
    """Check if a strategy is marked dead/error in existing results.

    Used by optimizer loops to skip dead strategies on re-runs.

    Args:
        results_file: Path to optimization_results.json.
        version: Strategy version string (e.g. "v1").

    Returns:
        True if the strategy should be skipped.
    """
    import json
    from pathlib import Path

    path = Path(results_file)
    if not path.exists():
        return False

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception):
        return False

    for entry in data:
        if entry.get("version") == version:
            status = entry.get("status", "")
            if status in ("dead", "error", "import_error"):
                return True
    return False
