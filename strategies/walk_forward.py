"""
strategies/walk_forward.py — Walk-Forward Validator
====================================================
Rolling walk-forward validation for any QBase strategy.

For each window:
  1. Load training data → run simplified Optuna optimization (coarse only)
  2. Load test data → backtest with optimized params
  3. Record test-period Sharpe + key metrics

Usage:
    python strategies/walk_forward.py \
        --strategy strategies/strong_trend/v12.py \
        --symbol AG \
        --train-years 5 --test-years 1 \
        --start 2015 --end 2026 \
        --freq daily

    # Quick mode (10 trials/window instead of 30)
    python strategies/walk_forward.py \
        --strategy strategies/strong_trend/v12.py \
        --symbol AG --quick
"""
import sys
import os
import json
import importlib
import importlib.util
import warnings
import argparse
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Setup paths
QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import optuna
from optuna.samplers import TPESampler

from strategies.optimizer_core import (
    auto_discover_params,
    suggest_params,
    create_strategy_with_params,
    composite_objective,
    run_single_backtest,
)
from config import get_data_dir


# =====================================================================
# Window Generation
# =====================================================================

def generate_windows(start_year, end_year, train_years, test_years):
    """Generate rolling walk-forward windows.

    Each window has a training period [train_start, train_end] and a test
    period [test_start, test_end].  Windows roll forward by ``test_years``
    each step.

    Args:
        start_year: first year of training data (int)
        end_year: last year of data (int, exclusive upper bound for test end)
        train_years: length of training window in years
        test_years: length of test window in years

    Returns:
        list of dicts with keys:
            window_id, train_start, train_end, test_start, test_end
    """
    windows = []
    window_id = 1
    train_start = start_year

    while True:
        train_end_year = train_start + train_years
        test_start_year = train_end_year
        test_end_year = test_start_year + test_years

        if test_end_year > end_year:
            break

        windows.append({
            "window_id": window_id,
            "train_start": f"{train_start}-01-01",
            "train_end": f"{train_end_year - 1}-12-31",
            "test_start": f"{test_start_year}-01-01",
            "test_end": f"{test_end_year - 1}-12-31",
        })

        window_id += 1
        train_start += test_years

    return windows


# =====================================================================
# Strategy Loading
# =====================================================================

def load_strategy_class_from_path(strategy_path):
    """Dynamically load a strategy class from a .py file path.

    Searches the module for a class whose name starts with a capital
    letter and has a ``warmup`` attribute (convention for AlphaForge
    strategies).

    Args:
        strategy_path: path to a .py strategy file

    Returns:
        (strategy_cls, strategy_name) tuple
    """
    path = Path(strategy_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the strategy class — look for a class with 'warmup' attribute
    candidates = []
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "warmup")
            and attr_name[0].isupper()
            and not attr_name.startswith("_")
        ):
            candidates.append((attr_name, obj))

    if not candidates:
        raise ValueError(
            f"No strategy class found in {path}. "
            "Expected a class with a 'warmup' attribute."
        )

    # Prefer class names containing "Strategy" or the longest name
    for name, cls in candidates:
        if "Strategy" in name or "strategy" in name:
            return cls, name

    # Fallback: return the last candidate (typically the main class)
    return candidates[-1][1], candidates[-1][0]


# =====================================================================
# Single-Window Optimization
# =====================================================================

def optimize_window(
    strategy_cls,
    param_specs,
    symbol,
    train_start,
    train_end,
    freq="daily",
    n_trials=30,
    seed=42,
    data_dir=None,
):
    """Run simplified coarse-only optimization for one walk-forward window.

    Uses TPESampler with a fixed seed for reproducibility.  Evaluates
    the strategy on a single symbol over the training period.

    Args:
        strategy_cls: the strategy class
        param_specs: parameter specs from auto_discover_params
        symbol: commodity symbol (e.g. "AG")
        train_start: training period start date string
        train_end: training period end date string
        freq: strategy frequency
        n_trials: number of Optuna trials (default 30)
        seed: random seed for TPE sampler
        data_dir: AlphaForge data directory

    Returns:
        dict with best_params and best_value, or None if optimization failed
    """
    if data_dir is None:
        data_dir = get_data_dir()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest_params(trial, param_specs)
        strategy = create_strategy_with_params(strategy_cls, params)
        r = run_single_backtest(
            strategy, symbol, train_start, train_end,
            freq=freq, data_dir=data_dir,
        )
        if r["sharpe"] <= -900:
            return -10.0
        return composite_objective([r], min_valid=1, freq=freq, scoring_mode="tanh")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_value <= -5.0:
        return None

    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
    }


# =====================================================================
# Walk-Forward Runner
# =====================================================================

def run_walk_forward(
    strategy_path,
    symbol,
    start_year=2015,
    end_year=2026,
    train_years=5,
    test_years=1,
    freq="daily",
    n_trials=30,
    verbose=True,
):
    """Run full walk-forward validation.

    Args:
        strategy_path: path to strategy .py file
        symbol: commodity symbol
        start_year: first year of data
        end_year: last year of data (exclusive)
        train_years: training window length
        test_years: test window length
        freq: strategy frequency
        n_trials: Optuna trials per window
        verbose: print progress

    Returns:
        dict with windows (list), summary, and metadata
    """
    # Load strategy
    strategy_cls, strategy_name = load_strategy_class_from_path(strategy_path)
    param_specs = auto_discover_params(strategy_cls)
    data_dir = get_data_dir()

    if not param_specs:
        raise ValueError(f"No tunable parameters found for {strategy_name}")

    # Generate windows
    windows = generate_windows(start_year, end_year, train_years, test_years)

    if not windows:
        raise ValueError(
            f"No valid windows for start={start_year}, end={end_year}, "
            f"train={train_years}y, test={test_years}y"
        )

    if verbose:
        strategy_stem = Path(strategy_path).stem
        print(f"\n{'='*65}")
        print(f"Walk-Forward Validation: {strategy_stem} on {symbol} ({freq})")
        print(f"  Windows: {len(windows)} | Trials/window: {n_trials}")
        print(f"  Params: {[s['name'] for s in param_specs]}")
        print(f"{'='*65}")

    # Run each window
    results = []
    for w in windows:
        wid = w["window_id"]
        seed = 42 + wid * 1000  # unique but reproducible seed per window

        if verbose:
            print(
                f"\n  Window {wid}: train {w['train_start'][:7]}~{w['train_end'][:7]}"
                f" | test {w['test_start'][:7]}~{w['test_end'][:7]}"
            )

        # Phase 1: Optimize on training data
        opt_result = optimize_window(
            strategy_cls, param_specs, symbol,
            w["train_start"], w["train_end"],
            freq=freq, n_trials=n_trials, seed=seed,
            data_dir=data_dir,
        )

        if opt_result is None:
            if verbose:
                print("    [SKIP] Optimization failed — insufficient data or no valid params")
            results.append({
                **w,
                "status": "optimization_failed",
                "train_score": None,
                "test_sharpe": None,
                "test_trades": None,
                "test_return": None,
                "test_max_dd": None,
                "best_params": None,
            })
            continue

        best_params = opt_result["best_params"]
        train_score = opt_result["best_value"]

        # Phase 2: Test with optimized params (locked, no changes)
        strategy = create_strategy_with_params(strategy_cls, best_params)
        test_result = run_single_backtest(
            strategy, symbol, w["test_start"], w["test_end"],
            freq=freq, data_dir=data_dir,
        )

        if test_result["sharpe"] <= -900:
            if verbose:
                print(f"    Train score: {train_score:.3f} | Test: FAILED (insufficient data)")
            results.append({
                **w,
                "status": "test_failed",
                "train_score": train_score,
                "test_sharpe": None,
                "test_trades": None,
                "test_return": None,
                "test_max_dd": None,
                "best_params": best_params,
            })
            continue

        test_sharpe = test_result["sharpe"]
        test_trades = test_result["n_trades"]
        test_return = test_result["total_return"]
        test_max_dd = test_result["max_drawdown"]

        if verbose:
            trades_str = str(test_trades) if test_trades is not None else "?"
            print(
                f"    Train score: {train_score:.3f} | "
                f"Test Sharpe: {test_sharpe:.2f} | "
                f"Trades: {trades_str}"
            )

        results.append({
            **w,
            "status": "ok",
            "train_score": train_score,
            "test_sharpe": test_sharpe,
            "test_trades": test_trades,
            "test_return": test_return,
            "test_max_dd": test_max_dd,
            "best_params": _serialize_params(best_params),
        })

    # Compute summary
    summary = _compute_summary(results)

    # Build output
    strategy_stem = Path(strategy_path).stem
    output = {
        "strategy": strategy_stem,
        "strategy_class": strategy_name,
        "symbol": symbol,
        "freq": freq,
        "start_year": start_year,
        "end_year": end_year,
        "train_years": train_years,
        "test_years": test_years,
        "n_trials": n_trials,
        "timestamp": datetime.now().isoformat(),
        "windows": results,
        "summary": summary,
    }

    if verbose:
        _print_summary_table(output)

    return output


def _serialize_params(params):
    """Convert param values to JSON-safe types."""
    if params is None:
        return None
    return {
        k: float(v) if isinstance(v, (np.floating, float)) else int(v)
        for k, v in params.items()
    }


def _compute_summary(results):
    """Compute summary statistics from walk-forward results."""
    ok_results = [r for r in results if r["status"] == "ok"]

    if not ok_results:
        return {
            "n_windows": len(results),
            "n_valid": 0,
            "mean_sharpe": None,
            "std_sharpe": None,
            "worst_sharpe": None,
            "best_sharpe": None,
            "win_rate": 0.0,
            "win_count": 0,
            "positive_windows": [],
            "negative_windows": [],
        }

    sharpes = [r["test_sharpe"] for r in ok_results]
    wins = [r for r in ok_results if r["test_sharpe"] > 0]
    losses = [r for r in ok_results if r["test_sharpe"] <= 0]

    return {
        "n_windows": len(results),
        "n_valid": len(ok_results),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "worst_sharpe": float(np.min(sharpes)),
        "best_sharpe": float(np.max(sharpes)),
        "win_rate": len(wins) / len(ok_results),
        "win_count": len(wins),
        "positive_windows": [r["window_id"] for r in wins],
        "negative_windows": [r["window_id"] for r in losses],
    }


# =====================================================================
# Display
# =====================================================================

def _print_summary_table(output):
    """Print a formatted walk-forward summary table."""
    results = output["windows"]
    summary = output["summary"]
    strategy = output["strategy"]
    symbol = output["symbol"]
    freq = output["freq"]

    header = f"Walk-Forward Validation: {strategy} on {symbol} ({freq})"
    width = 67

    print(f"\n{'='*width}")
    print(f" {header}")
    print(f"{'='*width}")
    print(
        f" {'Window':<8}| {'Train Period':<18}| {'Test Period':<13}"
        f"| {'Test Sharpe':>11} | {'Trades':>6}"
    )
    print(f"{'-'*width}")

    for r in results:
        wid = r["window_id"]
        train_period = f"{r['train_start'][:7]}~{r['train_end'][:7]}"
        test_period = f"{r['test_start'][:7]}~{r['test_end'][:7]}"

        if r["status"] != "ok":
            sharpe_str = f"{'SKIP':>11}"
            trades_str = f"{'—':>6}"
        else:
            sharpe_str = f"{r['test_sharpe']:>11.2f}"
            trades_str = (
                f"{r['test_trades']:>6}" if r["test_trades"] is not None else f"{'?':>6}"
            )

        print(f" {wid:<8}| {train_period:<18}| {test_period:<13}| {sharpe_str} | {trades_str}")

    print(f"{'-'*width}")

    if summary["n_valid"] > 0:
        print(f" Summary")
        print(
            f"   Mean Sharpe: {summary['mean_sharpe']:.2f}  |  "
            f"Std: {summary['std_sharpe']:.2f}  |  "
            f"Worst: {summary['worst_sharpe']:.2f}"
        )
        print(
            f"   Win Rate: {summary['win_count']}/{summary['n_valid']} "
            f"({summary['win_rate']:.0%})"
        )
    else:
        print(" Summary: No valid windows")

    print(f"{'='*width}")


# =====================================================================
# JSON Output
# =====================================================================

def save_results(output, output_dir=None):
    """Save walk-forward results to JSON.

    Default path: research_log/walk_forward/{strategy}_{symbol}.json

    Args:
        output: walk-forward result dict
        output_dir: override output directory

    Returns:
        str: path to saved file
    """
    if output_dir is None:
        output_dir = os.path.join(QBASE_ROOT, "research_log", "walk_forward")

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output['strategy']}_{output['symbol']}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return filepath


# =====================================================================
# CLI
# =====================================================================

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validator — rolling out-of-sample validation",
    )
    parser.add_argument(
        "--strategy", required=True,
        help="Path to strategy .py file (e.g. strategies/strong_trend/v12.py)",
    )
    parser.add_argument(
        "--symbol", required=True,
        help="Commodity symbol (e.g. AG, I)",
    )
    parser.add_argument("--train-years", type=int, default=5, help="Training window (years)")
    parser.add_argument("--test-years", type=int, default=1, help="Test window (years)")
    parser.add_argument("--start", type=int, default=2015, help="Start year")
    parser.add_argument("--end", type=int, default=2026, help="End year (exclusive)")
    parser.add_argument("--freq", default="daily", help="Strategy frequency")
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Optuna trials per window (default 30)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 trials per window for fast testing",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--output-dir", default=None, help="Override output directory")

    args = parser.parse_args(argv)

    if args.quick:
        args.trials = 10

    return args


def main(argv=None):
    """CLI entry point."""
    args = parse_args(argv)

    output = run_walk_forward(
        strategy_path=args.strategy,
        symbol=args.symbol,
        start_year=args.start,
        end_year=args.end,
        train_years=args.train_years,
        test_years=args.test_years,
        freq=args.freq,
        n_trials=args.trials,
        verbose=not args.quiet,
    )

    filepath = save_results(output, output_dir=args.output_dir)
    print(f"\nResults saved to {filepath}")

    return output


if __name__ == "__main__":
    main()
