"""
Boss Strategies Optimizer

Uses extended strong_trend training periods (±2 months).
Multi-symbol evaluation with composite scoring (0-10).

Usage:
    python optimizer.py --strategy v1 --trials 80
    python optimizer.py --strategy all --trials 80
    python optimizer.py --strategy v1 --multi-seed
"""

import sys
import json
import argparse
import importlib.util
import traceback
import time
import logging
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401
from config import get_data_dir

import numpy as np

from strategies.optimizer_core import (
    auto_discover_params, create_strategy_with_params,
    composite_objective, run_single_backtest,
    optimize_two_phase, optimize_multi_seed,
    narrow_param_space,
)
from strategies.boss.config import TRAINING_SYMBOLS, TRAINING_PERIODS

logging.getLogger("alphaforge").setLevel(logging.ERROR)
logging.getLogger("alphaforge.engine").setLevel(logging.ERROR)

BOSS_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BOSS_DIR / "optimization_results.json"


def load_strategy_class(version):
    filepath = BOSS_DIR / f"{version}.py"
    if not filepath.exists():
        raise FileNotFoundError(f"Strategy file not found: {filepath}")
    spec = importlib.util.spec_from_file_location(f"boss_{version}", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from alphaforge.strategy.base import TimeSeriesStrategy
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, TimeSeriesStrategy)
                and attr is not TimeSeriesStrategy):
            return attr
    raise ValueError(f"No TimeSeriesStrategy subclass found in {filepath}")


def optimize_single(version, n_trials=80, seed=42, probe_trials=5, multi_seed=False):
    try:
        strategy_cls = load_strategy_class(version)
        freq = getattr(strategy_cls, 'freq', 'daily')
        param_specs = auto_discover_params(strategy_cls)

        print(f"\n{'='*60}")
        print(f"Boss {version} | freq={freq} | trials={n_trials}")
        print(f"{'='*60}")

        if not param_specs:
            print(f"  WARNING: No tunable params for {version}")
            return None

        print(f"  Parameters: {[(p['name'], round(p['low'],2), round(p['high'],2)) for p in param_specs]}")

        t0 = time.time()
        data_dir = get_data_dir()

        def objective_fn(params):
            strategy = create_strategy_with_params(strategy_cls, params)
            results = []
            for symbol in TRAINING_SYMBOLS:
                start, end = TRAINING_PERIODS[symbol]
                r = run_single_backtest(strategy, symbol, start, end,
                                        freq=freq, data_dir=data_dir)
                if r['sharpe'] > -900:
                    results.append(r)
            if len(results) < 3:
                return -10.0
            return composite_objective(results, min_valid=3, freq=freq)

        if multi_seed:
            result = optimize_multi_seed(
                objective_fn, param_specs,
                coarse_trials=max(20, n_trials // 3),
                fine_trials=max(30, n_trials // 2),
                seeds=(42, 123, 456),
                study_name=f"boss_{version}",
                verbose=True, probe_trials=probe_trials,
            )
            elapsed = time.time() - t0
            output = {
                "version": version, "freq": freq,
                "best_params": result["best_params"],
                "best_score": result["best_value"],
                "n_trials": result.get("n_trials_total", result.get("n_trials")),
                "robustness": result.get("robustness"),
                "is_consistent": result.get("is_consistent"),
                "cross_seed_std": result.get("cross_seed_std"),
                "elapsed_seconds": round(elapsed, 1),
            }
        else:
            result = optimize_two_phase(
                objective_fn, param_specs,
                coarse_trials=max(20, n_trials // 3),
                fine_trials=max(30, n_trials // 2),
                seed=seed,
                study_name=f"boss_{version}",
                verbose=True, probe_trials=probe_trials,
            )
            elapsed = time.time() - t0
            output = {
                "version": version, "freq": freq,
                "best_params": result["best_params"],
                "best_score": result["best_value"],
                "n_trials": result["n_trials"],
                "robustness": result.get("robustness"),
                "elapsed_seconds": round(elapsed, 1),
            }

        print(f"  Score: {output['best_score']:.4f}/10")
        print(f"  Params: {output['best_params']}")
        print(f"  Time: {elapsed:.1f}s")
        return output

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {"version": version, "error": str(e)}


def save_results(results, filepath=None):
    if filepath is None:
        filepath = RESULTS_FILE
    existing = []
    if filepath.exists():
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing_map = {r["version"]: r for r in existing}
    for r in results:
        if r:
            existing_map[r["version"]] = r
    merged = sorted(existing_map.values(), key=lambda r: int(r["version"][1:]))
    with open(filepath, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nSaved to {filepath} ({len(merged)} strategies)")


def main():
    parser = argparse.ArgumentParser(description="Boss Strategy Optimizer")
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true")
    args = parser.parse_args()

    if args.strategy == "all":
        versions = [f.stem for f in sorted(BOSS_DIR.glob("v*.py"))]
    elif "," in args.strategy:
        versions = [v.strip() for v in args.strategy.split(",")]
    else:
        versions = [args.strategy]

    print(f"Boss Optimizer | {len(versions)} strategies | trials={args.trials}")
    print(f"Training: {TRAINING_SYMBOLS} (extended ±2 months)")

    results = []
    for ver in versions:
        r = optimize_single(ver, n_trials=args.trials, seed=args.seed,
                           multi_seed=args.multi_seed)
        results.append(r)

    save_results(results)

    valid = [r for r in results if r and "error" not in r]
    print(f"\n{'='*60}")
    print("BOSS OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for r in sorted(valid, key=lambda x: x.get("best_score", -999), reverse=True):
        tag = 'R' if r.get('robustness', {}).get('is_robust') else 'F'
        print(f"  {r['version']:>4s}: {r['best_score']:>5.2f}/10 [{tag}] freq={r['freq']}")


if __name__ == "__main__":
    main()
