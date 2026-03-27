"""
strategies/optimizer_core.py — Shared optimization infrastructure for QBase.

Provides:
1. Parameter auto-discovery from strategy class annotations
2. Composite objective function (Sharpe + consistency + drawdown penalty)
3. Two-phase optimization (coarse → fine)
4. Parameter robustness checking (plateau vs spike detection)
5. Multi-seed validation
6. Unified backtest runner with freq mapping and bar resampling

Used by:
- strategies/strong_trend/optimizer.py
- strategies/all_time/ag/optimizer.py
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
    """
    narrowed = []
    for spec in param_specs:
        name = spec["name"]
        orig_low, orig_high = spec["low"], spec["high"]
        if name in best_params:
            val = best_params[name]
            half_range = (orig_high - orig_low) * shrink_ratio / 2
            new_low = max(orig_low, val - half_range)
            new_high = min(orig_high, val + half_range)
            new_step = spec["step"] / 2 if spec["step"] else spec["step"]
            narrowed.append({**spec, "low": new_low, "high": new_high, "step": new_step})
        else:
            narrowed.append(spec)
    return narrowed


# =====================================================================
# 2. Composite Objective Function
# =====================================================================

# Minimum trade counts by frequency (from CLAUDE.md statistical significance standards)
MIN_TRADES_BY_FREQ = {
    "daily": 30,
    "4h": 50,
    "1h": 80,
    "60min": 80,
    "30min": 100,
    "15min": 150,
    "10min": 150,
    "5min": 150,
}


def _drawdown_penalty(mean_abs_dd):
    """Non-linear drawdown penalty with escalating tiers.

    Designed so that moderate drawdowns barely hurt, but large drawdowns
    become increasingly painful — matching real-world risk tolerance.

    Tiers:
        |MaxDD| <= 15%  →  0          (acceptable for trend strategies)
        15% < dd <= 25% →  0.3 per 1% over 15%   (light warning)
        25% < dd <= 35% →  above + 0.8 per 1% over 25%   (serious concern)
        dd > 35%        →  above + 2.0 per 1% over 35%   (deal-breaker)

    Examples:
        dd=10% → 0.0
        dd=20% → 0.015   (barely noticeable)
        dd=30% → 0.070   (meaningful reduction)
        dd=40% → 0.170   (heavy penalty, needs Sharpe>2 to compensate)
        dd=50% → 0.370   (almost certainly rejected)

    Returns:
        float: penalty value (always >= 0)
    """
    dd = mean_abs_dd  # already positive (abs)
    penalty = 0.0

    if dd <= 0.15:
        return 0.0

    # Tier 1: 15% - 25% (light)
    t1 = min(dd, 0.25) - 0.15
    penalty += 0.3 * t1

    if dd <= 0.25:
        return penalty

    # Tier 2: 25% - 35% (serious)
    t2 = min(dd, 0.35) - 0.25
    penalty += 0.8 * t2

    if dd <= 0.35:
        return penalty

    # Tier 3: > 35% (deal-breaker)
    t3 = dd - 0.35
    penalty += 2.0 * t3

    return penalty


def composite_objective(results, min_valid=1, freq=None):
    """Compute composite objective from backtest results.

    Multi-symbol formula (len(results) > 1):
        mean_sharpe + 0.3 * min_sharpe - drawdown_penalty(mean_abs_dd)

    Single-symbol formula (len(results) == 1):
        sharpe - drawdown_penalty(|max_dd|)

    The consistency bonus (0.3 * min_sharpe) only applies to multi-symbol
    evaluation, where it rewards parameter sets that work across ALL symbols.
    For single-symbol it would just scale Sharpe by 1.3x with no real meaning.

    Drawdown penalty is non-linear with escalating tiers:
        |MaxDD| <= 15%  →  no penalty
        15% - 25%       →  light (0.3 per 1%)
        25% - 35%       →  serious (0.8 per 1%)
        > 35%           →  deal-breaker (2.0 per 1%)

    Trade count filter: if ``freq`` is provided, results with fewer trades
    than the frequency-based threshold are excluded as statistically
    unreliable (e.g. daily needs >= 30 trades).

    Args:
        results: list of dicts, each with key 'sharpe' (required)
                 and optionally 'max_drawdown', 'n_trades'.
        min_valid: minimum number of valid results after filtering.
        freq: strategy frequency (e.g. 'daily', '4h') for trade count threshold.
              None disables the trade count filter.

    Returns:
        float: composite score, or -10.0 if insufficient valid results.
    """
    min_trades = MIN_TRADES_BY_FREQ.get(freq, 0) if freq else 0

    valid = []
    for r in results:
        if r.get("sharpe", -999) <= -900:
            continue
        # Trade count filter: skip results below statistical significance
        if min_trades > 0 and r.get("n_trades") is not None:
            if r["n_trades"] < min_trades:
                continue
        valid.append(r)

    if len(valid) < min_valid:
        return -10.0

    sharpes = [r["sharpe"] for r in valid]
    mean_sharpe = float(np.mean(sharpes))

    # Non-linear drawdown penalty (escalating tiers)
    dd_penalty = 0.0
    drawdowns = [r["max_drawdown"] for r in valid if r.get("max_drawdown") is not None]
    if drawdowns:
        mean_abs_dd = float(np.mean([abs(d) for d in drawdowns]))
        dd_penalty = _drawdown_penalty(mean_abs_dd)

    if len(valid) == 1:
        # Single-symbol: no consistency bonus (would just be 1.3x scaling)
        return mean_sharpe - dd_penalty

    # Multi-symbol: add consistency bonus rewarding cross-symbol robustness
    min_sharpe = float(np.min(sharpes))
    return mean_sharpe + 0.3 * min_sharpe - dd_penalty


# =====================================================================
# 3. Backtest Runner
# =====================================================================

def map_freq(freq):
    """Map strategy frequency to (AlphaForge load freq, resample factor).

    AlphaForge native: 1min, 5min, 10min, 15min, 30min, 60min, daily.
    Non-native frequencies are loaded at a base freq and resampled.

    Returns:
        (load_freq, resample_factor) where factor=1 means no resample.
    """
    NATIVE = {"1min", "5min", "10min", "15min", "30min", "60min", "daily"}
    if freq in NATIVE:
        return freq, 1

    RESAMPLE_MAP = {
        "1h": ("60min", 1),
        "20min": ("10min", 2),
        "4h": ("60min", 4),
    }
    if freq in RESAMPLE_MAP:
        return RESAMPLE_MAP[freq]

    return freq, 1


def resample_bars(bars, step):
    """Resample BarArray by grouping every ``step`` bars.

    E.g. step=4 converts 60min bars to 4h bars.
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


def run_single_backtest(
    strategy,
    symbol,
    start,
    end,
    freq="daily",
    data_dir=None,
    initial_capital=1_000_000,
    slippage_ticks=1.0,
):
    """Run a single backtest and return a result dict.

    Returns:
        dict with keys:
            sharpe (float): Sharpe ratio, or -999.0 on failure
            max_drawdown (float|None): Maximum drawdown (negative)
            n_trades (int|None): Total trade count
            total_return (float|None): Total return
    """
    FAIL = {
        "sharpe": -999.0,
        "max_drawdown": None,
        "n_trades": None,
        "total_return": None,
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

        return {
            "sharpe": sharpe,
            "max_drawdown": float(max_dd) if max_dd is not None else None,
            "n_trades": int(n_trades) if n_trades is not None else None,
            "total_return": float(total_return) if total_return is not None else None,
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
        return objective_fn(params)

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
        return objective_fn(params)

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
