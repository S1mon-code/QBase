"""
Portfolio Selection Stability Test
===================================
Tests whether the same strategies get selected when the returns data is
slightly perturbed (80% subsample without replacement).  This detects
overfit strategy selection — if small data changes flip the portfolio,
the selection is fragile.

Reuses existing backtest equity curves (no re-running backtests).
Only resamples the daily returns matrix and re-runs selection + HRP.

Usage:
    # From existing weights JSON (reads equity curves from individual backtests)
    python portfolio/stability_test.py \
        --weights strategies/strong_trend/portfolio/weights_ag.json \
        --n-runs 50 --subsample 0.8

    # From builder pipeline (pass strategy dir + optimization results)
    python portfolio/stability_test.py \
        --strategy-dir strategies/strong_trend \
        --symbol AG --start 2025-01-01 --end 2026-03-01 \
        --n-runs 50 --subsample 0.8
"""
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
from collections import defaultdict

from portfolio.builder import (
    build_returns_df,
    select_strategies,
    compute_shrunk_hrp_sharpe_weights,
    apply_weight_cap,
    calc_portfolio_sharpe_with_penalty,
)


# =========================================================================
# Core stability analysis
# =========================================================================

def subsample_returns(returns_df, frac=0.8, rng=None):
    """Randomly select `frac` of the trading days (without replacement).

    Args:
        returns_df: DataFrame with strategies as columns, dates as rows.
        frac: Fraction of rows to keep (0 < frac < 1).
        rng: numpy random Generator for reproducibility.

    Returns:
        Subsampled DataFrame with same columns, fewer rows.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_rows = len(returns_df)
    n_keep = max(1, int(n_rows * frac))
    idx = rng.choice(n_rows, size=n_keep, replace=False)
    idx.sort()
    return returns_df.iloc[idx]


def run_single_stability_iteration(
    returns_df,
    pool_versions,
    strategy_sharpes,
    penalty_weight=0.1,
    max_weight=0.20,
    frac=0.8,
    rng=None,
):
    """Run one iteration of stability test: subsample → select → HRP → weights.

    Args:
        returns_df: Full aligned returns DataFrame (all pool strategies).
        pool_versions: List of strategy version strings in the pool.
        strategy_sharpes: Dict mapping version → Sharpe ratio.
        penalty_weight: Drawdown overlap penalty weight.
        max_weight: Max single strategy weight cap.
        frac: Subsample fraction.
        rng: numpy random Generator.

    Returns:
        Dict with 'selected' (list) and 'weights' (dict version → weight).
        Returns None if iteration fails.
    """
    # Subsample returns
    sub_df = subsample_returns(returns_df, frac=frac, rng=rng)

    # Build returns dict from subsampled data
    sub_returns_dict = {}
    for col in sub_df.columns:
        series = sub_df[col].dropna()
        if len(series) > 5:
            sub_returns_dict[col] = series

    if len(sub_returns_dict) < 3:
        return None

    # Recompute individual Sharpes on subsampled data
    sub_sharpes = {}
    for v, series in sub_returns_dict.items():
        if series.std() > 0:
            sub_sharpes[v] = float(series.mean() / series.std() * np.sqrt(252))
        else:
            sub_sharpes[v] = 0.0

    # Build mock all_data for select_strategies
    mock_all_data = [{"version": v} for v in pool_versions if v in sub_returns_dict]

    try:
        selected, _ = select_strategies(mock_all_data, sub_returns_dict, penalty_weight)

        if not selected or len(selected) < 2:
            return None

        # HRP weighting on subsampled data
        weights, _, _, _ = compute_shrunk_hrp_sharpe_weights(
            selected, sub_returns_dict, sub_sharpes
        )
        weights = apply_weight_cap(weights, max_weight)

        return {"selected": selected, "weights": weights}
    except Exception:
        return None


def run_stability_test(
    returns_df,
    pool_versions,
    strategy_sharpes,
    n_runs=50,
    frac=0.8,
    penalty_weight=0.1,
    max_weight=0.20,
    seed=42,
    quiet=False,
):
    """Run full stability test: N iterations of subsample → select → weigh.

    Args:
        returns_df: Aligned returns DataFrame (strategies as columns).
        pool_versions: List of all strategy versions in pool.
        strategy_sharpes: Dict version → Sharpe.
        n_runs: Number of subsample iterations.
        frac: Fraction of days to keep per iteration.
        penalty_weight: DD overlap penalty.
        max_weight: Weight cap.
        seed: Random seed.
        quiet: Suppress per-iteration print output.

    Returns:
        Dict with stability results (selection_freq, weight_stats, classification).
    """
    rng = np.random.default_rng(seed)

    selection_counts = defaultdict(int)
    weight_samples = defaultdict(list)
    n_successful = 0

    # Redirect stdout during iterations to suppress builder noise
    import io
    original_stdout = sys.stdout

    for i in range(n_runs):
        if not quiet:
            # Show progress every 10 runs
            if (i + 1) % 10 == 0 or i == 0:
                sys.stdout = original_stdout
                print(f"  Stability iteration {i+1}/{n_runs}...")
                sys.stdout = io.StringIO()
            else:
                sys.stdout = io.StringIO()
        else:
            sys.stdout = io.StringIO()

        result = run_single_stability_iteration(
            returns_df=returns_df,
            pool_versions=pool_versions,
            strategy_sharpes=strategy_sharpes,
            penalty_weight=penalty_weight,
            max_weight=max_weight,
            frac=frac,
            rng=rng,
        )

        if result is not None:
            n_successful += 1
            for v in result["selected"]:
                selection_counts[v] += 1
            for v, w in result["weights"].items():
                weight_samples[v].append(w)

    # Restore stdout
    sys.stdout = original_stdout

    if n_successful == 0:
        return {
            "n_runs": n_runs,
            "n_successful": 0,
            "subsample_frac": frac,
            "strategies": [],
            "error": "All iterations failed",
        }

    # Compute stats per strategy
    strategies_stats = []
    all_versions = sorted(
        set(list(selection_counts.keys())),
        key=lambda v: selection_counts.get(v, 0),
        reverse=True,
    )

    for v in all_versions:
        count = selection_counts[v]
        freq = count / n_successful
        samples = weight_samples.get(v, [])

        if samples:
            avg_w = float(np.mean(samples))
            std_w = float(np.std(samples))
        else:
            avg_w = 0.0
            std_w = 0.0

        # Classification
        if freq > 0.80:
            classification = "CORE"
        elif freq >= 0.40:
            classification = "SATELLITE"
        else:
            classification = "EDGE"

        strategies_stats.append({
            "version": v,
            "selection_count": count,
            "selection_freq": round(freq, 4),
            "avg_weight": round(avg_w, 4),
            "std_weight": round(std_w, 4),
            "classification": classification,
        })

    return {
        "n_runs": n_runs,
        "n_successful": n_successful,
        "subsample_frac": frac,
        "seed": seed,
        "penalty_weight": penalty_weight,
        "max_weight": max_weight,
        "date": str(date.today()),
        "strategies": strategies_stats,
    }


def classify_strategy(freq):
    """Classify a strategy based on selection frequency.

    Args:
        freq: Selection frequency (0.0 to 1.0).

    Returns:
        'CORE' if freq > 0.80, 'SATELLITE' if 0.40-0.80, 'EDGE' if < 0.40.
    """
    if freq > 0.80:
        return "CORE"
    elif freq >= 0.40:
        return "SATELLITE"
    else:
        return "EDGE"


# =========================================================================
# Display
# =========================================================================

def print_stability_report(result):
    """Print formatted stability test report."""
    n_runs = result["n_successful"]
    frac = result["subsample_frac"]

    print(f"\n{'='*74}")
    print(f"  Portfolio Selection Stability Test ({n_runs} runs, {frac:.0%} subsample)")
    print(f"{'='*74}")

    strategies = result["strategies"]
    if not strategies:
        print("  No strategies selected in any iteration.")
        return

    # Table header
    print(f"  {'Strategy':<10s} {'Selected':<12s} {'Avg Weight':<18s} {'Classification':<14s}")
    print(f"  {'─'*10} {'─'*12} {'─'*18} {'─'*14}")

    for s in strategies:
        ver = s["version"]
        count = s["selection_count"]
        freq = s["selection_freq"]
        avg_w = s["avg_weight"]
        std_w = s["std_weight"]
        cls = s["classification"]

        freq_str = f"{count}/{n_runs} {freq:.0%}"
        weight_str = f"{avg_w:.2f} +/- {std_w:.2f}"

        print(f"  {ver:<10s} {freq_str:<12s} {weight_str:<18s} {cls:<14s}")

    # Summary by classification
    core = [s for s in strategies if s["classification"] == "CORE"]
    satellite = [s for s in strategies if s["classification"] == "SATELLITE"]
    edge = [s for s in strategies if s["classification"] == "EDGE"]

    print(f"\n  Classification:")
    print(f"    CORE (>80%):      {len(core)} strategies — high confidence, consistently selected")
    print(f"    SATELLITE (40-80%): {len(satellite)} strategies — moderate confidence")
    print(f"    EDGE (<40%):      {len(edge)} strategies — may be overfit selection")

    if core:
        total_core_weight = sum(s["avg_weight"] for s in core)
        print(f"\n  Core strategies carry {total_core_weight:.0%} avg total weight")


def save_stability_result(result, symbol, output_dir=None):
    """Save stability result to JSON.

    Args:
        result: Stability test result dict.
        symbol: Symbol name (for filename).
        output_dir: Output directory (default: research_log/robustness/).

    Returns:
        Path to saved file.
    """
    if output_dir is None:
        output_dir = Path(QBASE_ROOT) / "research_log" / "robustness"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stability_{symbol.lower()}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return str(output_path)


# =========================================================================
# Load data from existing weights JSON
# =========================================================================

def load_returns_from_weights(weights_path):
    """Load strategy equity curves and rebuild returns data from a weights JSON.

    This re-runs individual backtests to get equity curves.
    If daily_returns are stored in the JSON per-strategy, use those directly.

    Returns:
        (returns_df, pool_versions, strategy_sharpes, symbol)
    """
    with open(weights_path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    symbol = meta.get("symbol", "AG")
    start = meta.get("test_start", "2025-01-01")
    end = meta.get("test_end", "2026-03-01")
    strategy_dir_name = None

    # Try to determine strategy directory from the weights path
    weights_p = Path(weights_path)
    # Typically: strategies/<type>/portfolio/weights_<symbol>.json
    if weights_p.parent.name == "portfolio":
        strategy_dir_name = str(weights_p.parent.parent)

    # We need to re-run backtests to get equity curves
    # Import builder functions
    from portfolio.builder import collect_all_results, build_returns_df

    val_symbol = meta.get("validation_symbol", None)
    val_start = meta.get("validation_start", None)
    val_end = meta.get("validation_end", None)

    if val_symbol == "":
        val_symbol = None

    results_file = None
    if strategy_dir_name:
        rf = Path(strategy_dir_name) / "optimization_results.json"
        if rf.exists():
            results_file = str(rf)

    all_data = collect_all_results(
        symbol, start, end,
        val_symbol, val_start, val_end,
        strategy_dir=strategy_dir_name,
        results_file=results_file,
    )

    returns_dict = build_returns_df(all_data)
    pool_versions = [d["version"] for d in all_data if d["version"] in returns_dict]

    # Build aligned DataFrame
    returns_df = pd.DataFrame(returns_dict).fillna(0)

    # Strategy Sharpes
    strategy_sharpes = {}
    for d in all_data:
        if d["primary_sharpe"] is not None:
            strategy_sharpes[d["version"]] = d["primary_sharpe"]

    return returns_df, pool_versions, strategy_sharpes, symbol


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Portfolio Selection Stability Test"
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to existing weights JSON (will re-run backtests to get equity curves)"
    )
    parser.add_argument(
        "--strategy-dir", default=None,
        help="Strategy directory (alternative to --weights)"
    )
    parser.add_argument("--symbol", default="AG", help="Symbol (default: AG)")
    parser.add_argument("--start", default="2025-01-01", help="Start date")
    parser.add_argument("--end", default="2026-03-01", help="End date")
    parser.add_argument("--results-file", default=None, help="Optimization results JSON")
    parser.add_argument("--n-runs", type=int, default=50, help="Number of subsample iterations (default: 50)")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample fraction (default: 0.8)")
    parser.add_argument("--penalty-weight", type=float, default=0.1, help="DD overlap penalty (default: 0.1)")
    parser.add_argument("--max-weight", type=float, default=0.20, help="Max weight cap (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default=None, help="Output directory for JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*74}")
    print(f"  PORTFOLIO SELECTION STABILITY TEST")
    print(f"  {args.n_runs} iterations, {args.subsample:.0%} subsample, seed={args.seed}")
    print(f"{'='*74}\n")

    if args.weights:
        print(f"Loading from weights: {args.weights}")
        returns_df, pool_versions, strategy_sharpes, symbol = load_returns_from_weights(
            args.weights
        )
    elif args.strategy_dir:
        from portfolio.builder import collect_all_results, build_returns_df
        print(f"Running backtests for {args.symbol} from {args.strategy_dir}")
        all_data = collect_all_results(
            args.symbol, args.start, args.end,
            None, None, None,
            strategy_dir=args.strategy_dir,
            results_file=args.results_file,
        )
        returns_dict = build_returns_df(all_data)
        pool_versions = [d["version"] for d in all_data if d["version"] in returns_dict]
        returns_df = pd.DataFrame(returns_dict).fillna(0)
        strategy_sharpes = {}
        for d in all_data:
            if d["primary_sharpe"] is not None:
                strategy_sharpes[d["version"]] = d["primary_sharpe"]
        symbol = args.symbol.upper()
    else:
        print("ERROR: Must provide either --weights or --strategy-dir")
        sys.exit(1)

    print(f"\nPool: {len(pool_versions)} strategies, {len(returns_df)} trading days")
    print(f"Running stability test...\n")

    result = run_stability_test(
        returns_df=returns_df,
        pool_versions=pool_versions,
        strategy_sharpes=strategy_sharpes,
        n_runs=args.n_runs,
        frac=args.subsample,
        penalty_weight=args.penalty_weight,
        max_weight=args.max_weight,
        seed=args.seed,
    )

    # Add symbol to result
    result["symbol"] = symbol

    # Print report
    print_stability_report(result)

    # Save JSON
    output_path = save_stability_result(result, symbol, output_dir=args.output_dir)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
