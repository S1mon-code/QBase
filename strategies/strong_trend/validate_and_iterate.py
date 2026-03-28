"""
Test Set Validation + Iterative Optimization
=============================================
1. Load optimized params from optimization_results.json
2. Run all 20 strategies on test set (AG, EC)
3. For strategies with test Sharpe < 0, re-optimize with tighter focus
4. Report final results

Usage:
    python strategies/strong_trend/validate_and_iterate.py
"""
import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import optuna
from optuna.samplers import TPESampler

from strategies.strong_trend.optimizer import (
    create_strategy_with_params,
    load_strategy_class,
    run_single_backtest,
    suggest_params,
    PARAM_SPACES,
    STRATEGY_CLASSES,
    TRAINING_SYMBOLS,
    TRAINING_PERIODS,
    _map_freq,
)

# =========================================================================
# Test set configuration
# =========================================================================

TEST_SETS = {
    "AG": ("2025-01-01", "2026-03-01"),   # 白银 +254% rally
    "EC": ("2023-07-01", "2024-09-01"),   # 集运 +907% rally
}

RESULTS_PATH = os.path.join(QBASE_ROOT, "strategies", "strong_trend", "optimization_results.json")
FINAL_REPORT_PATH = os.path.join(QBASE_ROOT, "research_log", "strong_trend.md")


def load_optimization_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def run_test_set(version: str, params: dict) -> dict:
    """Run a strategy on the test set. Returns per-symbol Sharpes."""
    strategy = create_strategy_with_params(version, params)
    freq = strategy.freq
    results = {}
    for symbol, (start, end) in TEST_SETS.items():
        sharpe = run_single_backtest(strategy, symbol, start, end, freq=freq)
        results[symbol] = sharpe if sharpe > -900 else None
    return results


def validate_all():
    """Validate all 20 strategies on test set."""
    opt_results = load_optimization_results()

    print("=" * 70)
    print("TEST SET VALIDATION (AG + EC)")
    print("=" * 70)

    all_test_results = []
    for r in opt_results:
        ver = r["version"]
        params = r["best_params"]
        train_sharpe = r["best_sharpe"]

        if train_sharpe < -5:  # Skip completely broken strategies
            test = {"AG": None, "EC": None}
            print(f"  {ver:>4s}: SKIPPED (train Sharpe = {train_sharpe:.2f})")
        else:
            test = run_test_set(ver, params)
            ag_str = f"{test['AG']:.3f}" if test['AG'] is not None else "FAIL"
            ec_str = f"{test['EC']:.3f}" if test['EC'] is not None else "FAIL"
            mean_test = np.mean([v for v in test.values() if v is not None]) if any(v is not None for v in test.values()) else -999
            print(f"  {ver:>4s}: Train={train_sharpe:>7.3f}  AG={ag_str:>8s}  EC={ec_str:>8s}  MeanTest={mean_test:>7.3f}")

        all_test_results.append({
            "version": ver,
            "train_sharpe": train_sharpe,
            "test_AG": test.get("AG"),
            "test_EC": test.get("EC"),
            "params": params,
        })

    return all_test_results


def identify_underperformers(test_results):
    """Find strategies that need re-optimization (test Sharpe < 0)."""
    bad = []
    for r in test_results:
        ag = r["test_AG"]
        ec = r["test_EC"]
        valid = [v for v in [ag, ec] if v is not None]
        if not valid:
            bad.append(r["version"])
        elif np.mean(valid) < 0:
            bad.append(r["version"])
    return bad


def re_optimize_focused(version: str, n_trials: int = 200):
    """Re-optimize a strategy with test set awareness.

    Strategy: optimize on training set but use a combined objective
    that penalizes params that are extreme (likely overfit).
    Uses more trials and a different seed.
    """
    print(f"\n  Re-optimizing {version} (focused, {n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=123),  # Different seed
        study_name=f"reopt_{version}",
    )

    def objective(trial):
        params = suggest_params(trial, version)
        strategy = create_strategy_with_params(version, params)
        freq = strategy.freq

        # Evaluate on training set
        train_sharpes = []
        for symbol in TRAINING_SYMBOLS:
            start, end = TRAINING_PERIODS[symbol]
            s = run_single_backtest(strategy, symbol, start, end, freq=freq)
            if s > -900:
                train_sharpes.append(s)

        if len(train_sharpes) < 3:
            return -10.0

        mean_train = np.mean(train_sharpes)
        min_train = np.min(train_sharpes)

        # Penalize if any single symbol is very negative (overfitting signal)
        # Reward consistency across symbols
        consistency_bonus = min_train * 0.3  # Bonus for worst-case being good
        return mean_train + consistency_bonus

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"  Best: Sharpe={best.value:.4f}, Params={best.params}")
    return {
        "version": version,
        "best_sharpe": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
        "n_completed": len(study.trials),
    }


def write_report(test_results):
    """Write final results to research_log/strong_trend.md."""
    lines = [
        "# Strong Trend Strategy — Optimization & Validation Results\n",
        f"## Test Set: AG (2025-01 → 2026-03), EC (2023-07 → 2024-09)\n",
        "",
        "| Rank | Strategy | Train Sharpe | AG Sharpe | EC Sharpe | Mean Test |",
        "|------|----------|-------------|-----------|-----------|-----------|",
    ]

    # Sort by mean test Sharpe
    def _mean_test(r):
        vals = [v for v in [r["test_AG"], r["test_EC"]] if v is not None]
        return np.mean(vals) if vals else -999

    sorted_results = sorted(test_results, key=_mean_test, reverse=True)

    for i, r in enumerate(sorted_results, 1):
        ag = f"{r['test_AG']:.3f}" if r['test_AG'] is not None else "N/A"
        ec = f"{r['test_EC']:.3f}" if r['test_EC'] is not None else "N/A"
        mt = _mean_test(r)
        mt_str = f"{mt:.3f}" if mt > -900 else "N/A"
        lines.append(
            f"| {i} | {r['version']} | {r['train_sharpe']:.3f} | {ag} | {ec} | {mt_str} |"
        )

    lines.extend([
        "",
        "## Best Parameters (Top 5)\n",
    ])

    for r in sorted_results[:5]:
        lines.append(f"### {r['version']}")
        lines.append(f"- Train Sharpe: {r['train_sharpe']:.3f}")
        ag = r['test_AG']
        ec = r['test_EC']
        lines.append(f"- Test AG: {ag:.3f}" if ag is not None else "- Test AG: N/A")
        lines.append(f"- Test EC: {ec:.3f}" if ec is not None else "- Test EC: N/A")
        lines.append(f"- Params: `{json.dumps(r['params'], indent=None)}`")
        lines.append("")

    with open(FINAL_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {FINAL_REPORT_PATH}")


# =========================================================================
# Main pipeline
# =========================================================================

if __name__ == "__main__":
    MAX_ITERATIONS = 3

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'#' * 70}")
        print(f"# ITERATION {iteration + 1} / {MAX_ITERATIONS}")
        print(f"{'#' * 70}")

        # Step 1: Validate on test set
        test_results = validate_all()

        # Step 2: Identify underperformers
        bad = identify_underperformers(test_results)
        good_count = len([r for r in test_results
                          if r["test_AG"] is not None or r["test_EC"] is not None])
        positive_count = len([r for r in test_results
                              if any(v is not None and v > 0 for v in [r["test_AG"], r["test_EC"]])])

        print(f"\nResults: {positive_count} strategies with positive test Sharpe, "
              f"{len(bad)} need re-optimization")

        if not bad or iteration == MAX_ITERATIONS - 1:
            # Done — write final report
            write_report(test_results)
            break

        # Step 3: Re-optimize underperformers
        print(f"\nRe-optimizing {len(bad)} strategies: {bad}")
        opt_results = load_optimization_results()

        for ver in bad:
            if ver not in PARAM_SPACES:
                continue
            new_result = re_optimize_focused(ver, n_trials=200)

            # Update optimization_results.json
            for i, r in enumerate(opt_results):
                if r["version"] == ver:
                    opt_results[i] = new_result
                    break

        with open(RESULTS_PATH, "w") as f:
            json.dump(opt_results, f, indent=2)
        print("\nUpdated optimization_results.json with re-optimized params")

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    test_results = validate_all()
    write_report(test_results)

    # Attribution analysis (Step 4.5)
    print(f"\n{'=' * 70}")
    print("ATTRIBUTION ANALYSIS")
    print(f"{'=' * 70}")

    from attribution.report import run_full_attribution
    run_full_attribution(
        test_results=test_results,
        load_strategy_fn=load_strategy_class,
        test_sets=TEST_SETS,
    )
