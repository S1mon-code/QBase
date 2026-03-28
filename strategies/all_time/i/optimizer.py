"""
All-Time Iron Ore (I) Strategy Optimizer

Usage:
    python optimizer.py --strategy v1 --trials 50 --phase coarse
    python optimizer.py --strategy v1 --trials 100 --phase fine
    python optimizer.py --strategy v1-v50 --trials 50 --phase coarse
    python optimizer.py --strategy all --trials 50 --phase coarse
    python optimizer.py --strategy v1 --trials 50 --multi-seed
"""

import sys
import json
import argparse
import importlib.util
import traceback
import time
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401
from config import get_data_dir
import logging

import numpy as np

from strategies.optimizer_core import (
    auto_discover_params, create_strategy_with_params,
    composite_objective, run_single_backtest,
    optimize_two_phase, optimize_multi_seed,
    narrow_param_space,
)

logging.getLogger("alphaforge").setLevel(logging.ERROR)
logging.getLogger("alphaforge.engine").setLevel(logging.ERROR)

I_DIR = Path(__file__).resolve().parent
RESULTS_FILE = I_DIR / "optimization_results.json"
COARSE_RESULTS_FILE = I_DIR / "optimization_coarse.json"

TRAIN_END = "2021-12-31"
SYMBOL = "I"
DEFAULT_CAPITAL = 1_000_000


def load_strategy_class(version: str):
    filepath = I_DIR / f"{version}.py"
    if not filepath.exists():
        raise FileNotFoundError(f"Strategy file not found: {filepath}")
    spec = importlib.util.spec_from_file_location(f"strategy_{version}", filepath)
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


def get_strategy_freq(strategy_cls):
    return getattr(strategy_cls, 'freq', 'daily')


def optimize_single(version, n_trials=30, phase="coarse",
                    seed=42, probe_trials=5, multi_seed=False):
    try:
        strategy_cls = load_strategy_class(version)
        freq = get_strategy_freq(strategy_cls)
        param_specs = auto_discover_params(strategy_cls)

        print(f"\n{'='*60}")
        print(f"Optimizing {version} | freq={freq} | phase={phase} | trials={n_trials}")
        print(f"{'='*60}")

        if not param_specs:
            print(f"  WARNING: No tunable params for {version}")
            return None

        print(f"  Parameters: {[(p['name'], p['low'], p['high']) for p in param_specs]}")

        t0 = time.time()
        data_dir = get_data_dir()

        def objective_fn(params, scoring_mode="tanh"):
            strategy = create_strategy_with_params(strategy_cls, params)
            r = run_single_backtest(strategy, SYMBOL, start=None, end=TRAIN_END,
                                    freq=freq, data_dir=data_dir,
                                    initial_capital=DEFAULT_CAPITAL)
            return composite_objective([r], min_valid=1, freq=freq, scoring_mode=scoring_mode)

        if phase == "fine":
            coarse = load_coarse_results()
            coarse_best = coarse.get(version, {}).get("best_params", {})
            if coarse_best:
                param_specs = narrow_param_space(param_specs, coarse_best)
                print(f"  Fine-tuning around coarse best: {coarse_best}")
            else:
                print(f"  No coarse results for {version}, using full range")

        if multi_seed:
            result = optimize_multi_seed(
                objective_fn, param_specs,
                coarse_trials=max(15, n_trials // 2),
                fine_trials=max(15, n_trials // 2),
                seeds=(seed, seed + 100, seed + 200),
                study_name=f"i_alltime_{version}",
                verbose=True, probe_trials=probe_trials,
            )
            elapsed = time.time() - t0
            output = {
                "version": version, "phase": phase, "freq": freq,
                "best_params": result["best_params"],
                "best_sharpe": result["best_value"],
                "n_trials": result["n_trials_total"],
                "robustness": result.get("robustness"),
                "early_stopped": result.get("early_stopped", False),
                "multi_seed": True,
                "cross_seed_std": result.get("cross_seed_std"),
                "is_consistent": result.get("is_consistent"),
                "elapsed_seconds": round(elapsed, 1),
            }
        else:
            result = optimize_two_phase(
                objective_fn, param_specs,
                coarse_trials=max(15, n_trials // 2),
                fine_trials=max(15, n_trials // 2),
                seed=seed, study_name=f"i_alltime_{version}",
                verbose=True, probe_trials=probe_trials,
            )
            elapsed = time.time() - t0
            output = {
                "version": version, "phase": phase, "freq": freq,
                "best_params": result["best_params"],
                "best_sharpe": result["best_value"],
                "n_trials": result["n_trials"],
                "robustness": result.get("robustness"),
                "early_stopped": result.get("early_stopped", False),
                "elapsed_seconds": round(elapsed, 1),
            }

        print(f"  Best Sharpe: {output['best_sharpe']:.4f}")
        print(f"  Best params: {output['best_params']}")
        print(f"  Time: {elapsed:.1f}s")
        return output

    except Exception as e:
        print(f"  ERROR optimizing {version}: {e}")
        traceback.print_exc()
        return {"version": version, "phase": phase, "error": str(e)}


def load_coarse_results():
    if COARSE_RESULTS_FILE.exists():
        with open(COARSE_RESULTS_FILE) as f:
            data = json.load(f)
        return {r["version"]: r for r in data if "error" not in r}
    return {}


def save_results(results, filepath):
    existing = []
    if filepath.exists():
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, Exception):
            existing = []
    existing_map = {r["version"]: r for r in existing}
    for r in results:
        existing_map[r["version"]] = r
    merged = sorted(existing_map.values(), key=lambda r: int(r["version"][1:]))
    with open(filepath, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nResults saved to {filepath} ({len(merged)} total, {len(results)} new)")


def parse_strategy_range(s):
    if s == "all":
        files = sorted(I_DIR.glob("v*.py"))
        return [f.stem for f in files]
    if "-" in s and s.count("-") == 1:
        parts = s.split("-")
        start = int(parts[0].replace("v", ""))
        end = int(parts[1].replace("v", ""))
        return [f"v{i}" for i in range(start, end + 1)]
    if "," in s:
        return [v.strip() for v in s.split(",")]
    return [s]


def main():
    parser = argparse.ArgumentParser(description="All-Time Iron Ore Strategy Optimizer")
    parser.add_argument("--strategy", required=True, help="v1, v1-v50, all")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--phase", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true")
    args = parser.parse_args()

    versions = parse_strategy_range(args.strategy)
    print(f"Optimizing {len(versions)} strategies: {versions[0]}...{versions[-1]}")
    print(f"Phase: {args.phase} | Trials: {args.trials} | Symbol: {SYMBOL}")
    print(f"Training period: start ~ {TRAIN_END}")

    result_file = COARSE_RESULTS_FILE if args.phase == "coarse" else RESULTS_FILE
    already_done = set()
    if result_file.exists():
        try:
            with open(result_file) as f:
                for r in json.load(f):
                    already_done.add(r["version"])
        except (json.JSONDecodeError, Exception):
            pass

    results = []
    for version in versions:
        if not (I_DIR / f"{version}.py").exists():
            print(f"  Skipping {version}: file not found")
            continue
        if version in already_done:
            print(f"  Skipping {version}: already optimized")
            continue
        r = optimize_single(version, n_trials=args.trials, phase=args.phase,
                           seed=args.seed, multi_seed=args.multi_seed)
        if r:
            results.append(r)

    if args.phase == "coarse":
        save_results(results, COARSE_RESULTS_FILE)
    else:
        save_results(results, RESULTS_FILE)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SUMMARY ({args.phase})")
    print(f"{'='*60}")
    print(f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}")
    if successful:
        sharpes = [r["best_sharpe"] for r in successful]
        print(f"Sharpe range: [{min(sharpes):.4f}, {max(sharpes):.4f}]")
        print(f"Sharpe mean: {np.mean(sharpes):.4f} | median: {np.median(sharpes):.4f}")


if __name__ == "__main__":
    main()
