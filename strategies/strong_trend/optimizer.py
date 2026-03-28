"""
Optuna Optimizer for Strong Trend Strategies
=============================================
Optimizes each strategy's parameters on training set commodities.
Goal: maximize composite objective (Sharpe + consistency + drawdown penalty)
across multiple symbols to avoid overfitting.

Uses shared optimizer_core for:
- Auto-discovery of tunable parameters (no hardcoded PARAM_SPACES)
- Composite objective function
- Two-phase optimization (coarse -> fine)
- Multi-seed validation with robustness checking

Usage:
    python strategies/strong_trend/optimizer.py --strategy v1 --trials 80
    python strategies/strong_trend/optimizer.py --strategy all --trials 100
    python strategies/strong_trend/optimizer.py --strategy all --trials 100 --multi-seed
"""
import sys
import os
from pathlib import Path
import importlib
import json
import warnings

warnings.filterwarnings("ignore")

# Setup paths
QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

from strategies.optimizer_core import (
    auto_discover_params,
    suggest_params,
    create_strategy_with_params,
    composite_objective,
    run_single_backtest,
    optimize_two_phase,
    optimize_multi_seed,
    build_result_entry,
    detect_strategy_status,
    is_strategy_dead,
)

# =========================================================================
# Configuration
# =========================================================================

DATA_DIR = str(Path(conftest._af_path) / "data")

# Training symbols + periods (from RALLIES.md, excluding AG and EC)
TRAINING_SYMBOLS = ["J", "ZC", "JM", "I", "NI", "SA"]

# Wider date ranges (include pre-rally + rally + post-rally for realistic testing)
TRAINING_PERIODS = {
    "J":  ("2015-06-01", "2017-06-01"),   # 焦炭 supply-side reform rally
    "ZC": ("2020-06-01", "2022-06-01"),   # 动力煤 energy crisis
    "JM": ("2020-06-01", "2022-06-01"),   # 焦煤 carbon neutral
    "I":  ("2015-06-01", "2017-06-01"),   # 铁矿石 supply-side reform
    "NI": ("2021-01-01", "2022-09-01"),   # 镍 LME squeeze
    "SA": ("2022-06-01", "2024-06-01"),   # 纯碱 PV demand
}

# Strategy class names mapping
STRATEGY_CLASSES = {f"v{i}": f"StrongTrendV{i}" for i in range(1, 51)}
STRATEGY_CLASSES["v3"] = "DonchianADXChandelierStrategy"  # v3 has non-standard name


# =========================================================================
# Core functions
# =========================================================================

def load_strategy_class(version: str):
    """Dynamically import and return the strategy class."""
    mod = importlib.import_module(f"strategies.strong_trend.{version}")
    class_name = STRATEGY_CLASSES[version]
    return getattr(mod, class_name)


def evaluate_strategy(version, params, data_dir, scoring_mode="tanh"):
    """Evaluate strategy across all training symbols with composite objective."""
    strategy_cls = load_strategy_class(version)
    strategy = create_strategy_with_params(strategy_cls, params)
    freq = strategy.freq

    results = []
    for symbol in TRAINING_SYMBOLS:
        start, end = TRAINING_PERIODS[symbol]
        r = run_single_backtest(strategy, symbol, start, end, freq=freq, data_dir=data_dir)
        if r['sharpe'] > -900:
            results.append(r)

    if len(results) < 3:  # Need at least 3 valid symbols
        return -10.0
    return composite_objective(results, min_valid=3, freq=freq, scoring_mode=scoring_mode)


def optimize_strategy(version, n_trials=80, verbose=True, use_multi_seed=False):
    """Optimize a single strategy version."""
    import time

    # Skip dead strategies
    results_path = os.path.join(
        QBASE_ROOT, "strategies", "strong_trend", "optimization_results.json"
    )
    if is_strategy_dead(results_path, version):
        if verbose:
            print(f"  Skipping {version}: marked dead/error in results")
        return None

    try:
        strategy_cls = load_strategy_class(version)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  ERROR: {version} import failed: {e}")
        return build_result_entry(
            version=version, freq="daily", best_params={},
            sharpe=-999.0, score=-999.0, status="import_error",
        )
    except Exception as e:
        print(f"  ERROR: {version} load failed: {e}")
        return build_result_entry(
            version=version, freq="daily", best_params={},
            sharpe=-999.0, score=-999.0, status="error",
        )

    freq = getattr(strategy_cls, 'freq', 'daily')
    param_specs = auto_discover_params(strategy_cls)

    if not param_specs:
        print(f"  WARNING: No tunable params for {version}")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing {version} ({STRATEGY_CLASSES[version]})")
        print(f"  Params: {[s['name'] for s in param_specs]}")
        print(f"  Trials: {n_trials}, Multi-seed: {use_multi_seed}")
        print(f"  Training symbols: {TRAINING_SYMBOLS}")
        print(f"{'='*60}")

    data_dir = DATA_DIR
    t0 = time.time()

    def objective_fn(params, scoring_mode="tanh"):
        return evaluate_strategy(version, params, data_dir, scoring_mode=scoring_mode)

    if use_multi_seed:
        result = optimize_multi_seed(
            objective_fn, param_specs,
            coarse_trials=max(20, n_trials // 4),
            fine_trials=max(30, n_trials // 3),
            seeds=(42, 123, 456),
            verbose=verbose,
            study_name=f"strong_trend_{version}",
        )
    else:
        result = optimize_two_phase(
            objective_fn, param_specs,
            coarse_trials=max(20, n_trials // 3),
            fine_trials=max(30, n_trials // 2),
            seed=42,
            verbose=verbose,
            study_name=f"strong_trend_{version}",
        )

    elapsed = time.time() - t0
    best_value = result["best_value"]
    n_total = result.get("n_trials_total", result.get("n_trials", n_trials))
    phase = result.get("phase", "two_phase")

    # best_value is the composite score (0-10 scale)
    output = build_result_entry(
        version=version,
        freq=freq,
        best_params=result["best_params"],
        sharpe=best_value,      # legacy: composite score stored as best_sharpe
        score=best_value,       # also store as best_score
        n_trials=n_total,
        phase=phase,
        robustness=result.get("robustness"),
        elapsed=elapsed,
        is_consistent=result.get("is_consistent"),
    )
    output["status"] = detect_strategy_status(output)
    return output


def save_results(results: list, output_path: str = None):
    """Save optimization results to JSON."""
    if output_path is None:
        output_path = os.path.join(
            QBASE_ROOT, "strategies", "strong_trend", "optimization_results.json"
        )

    # Filter out None results and make JSON-serializable
    clean = []
    for r in results:
        if r is None:
            continue
        entry = dict(r)
        # robustness dict may contain numpy types
        if entry.get("robustness") and isinstance(entry["robustness"], dict):
            entry["robustness"] = {
                k: float(v) if hasattr(v, '__float__') else v
                for k, v in entry["robustness"].items()
            }
        clean.append(entry)

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {output_path}")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize strong trend strategies")
    parser.add_argument(
        "--strategy", default="v1",
        help="Strategy version (v1-v50) or 'all' for all strategies"
    )
    parser.add_argument("--trials", type=int, default=80, help="Optuna trials per strategy")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Enable multi-seed validation (3 seeds, 3x compute — for final top candidates)"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    use_multi_seed = args.multi_seed

    if args.strategy == "all":
        versions = [f"v{i}" for i in range(1, 51)]
    else:
        versions = [args.strategy]

    all_results = []
    for ver in versions:
        result = optimize_strategy(ver, n_trials=args.trials, verbose=verbose,
                                   use_multi_seed=use_multi_seed)
        all_results.append(result)
        if result:
            tag = ""
            if result.get("is_consistent") is not None:
                tag = " [consistent]" if result["is_consistent"] else " [inconsistent]"
            print(f"  {ver}: best Sharpe = {result['best_sharpe']:.4f}{tag}")
        else:
            print(f"  {ver}: SKIPPED (no tunable params)")

    save_results(all_results)

    # Summary
    valid = [r for r in all_results if r is not None]
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for r in sorted(valid, key=lambda x: x["best_sharpe"], reverse=True):
        robust_tag = ""
        if r.get("robustness") and isinstance(r["robustness"], dict):
            robust_tag = " [R]" if r["robustness"].get("is_robust") else " [S]"
        consist_tag = ""
        if r.get("is_consistent") is not None:
            consist_tag = " C" if r["is_consistent"] else " I"
        print(f"  {r['version']:>4s}: Sharpe = {r['best_sharpe']:>8.4f}"
              f"{robust_tag}{consist_tag}  params = {r['best_params']}")
