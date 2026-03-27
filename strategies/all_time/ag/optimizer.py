"""
All-Time AG 策略优化器

用法:
    # 优化单个策略（粗调）
    python optimizer.py --strategy v1 --trials 50 --phase coarse

    # 优化单个策略（精调，基于粗调结果）
    python optimizer.py --strategy v1 --trials 100 --phase fine

    # 批量优化（粗调）
    python optimizer.py --strategy v1-v50 --trials 50 --phase coarse

    # 优化全部
    python optimizer.py --strategy all --trials 50 --phase coarse
"""

import sys
import os
import json
import argparse
import importlib.util
import traceback
from pathlib import Path

# Setup paths
QBASE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401
from config import get_data_dir
import logging

import numpy as np

# Suppress noisy AlphaForge order rejection logs
logging.getLogger("alphaforge").setLevel(logging.ERROR)
logging.getLogger("alphaforge.engine").setLevel(logging.ERROR)

AG_DIR = Path(__file__).resolve().parent
RESULTS_FILE = AG_DIR / "optimization_results.json"
COARSE_RESULTS_FILE = AG_DIR / "optimization_coarse.json"

# Training set: pre-2022
TRAIN_END = "2021-12-31"
SYMBOL = "AG"
DEFAULT_CAPITAL = 1_000_000


def load_strategy_class(version: str):
    """Load strategy class from vN.py file."""
    filepath = AG_DIR / f"{version}.py"
    if not filepath.exists():
        raise FileNotFoundError(f"Strategy file not found: {filepath}")

    spec = importlib.util.spec_from_file_location(f"strategy_{version}", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the strategy class (subclass of TimeSeriesStrategy)
    from alphaforge.strategy.base import TimeSeriesStrategy
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, TimeSeriesStrategy)
                and attr is not TimeSeriesStrategy):
            return attr
    raise ValueError(f"No TimeSeriesStrategy subclass found in {filepath}")


def get_strategy_freq(strategy_cls):
    """Get strategy frequency."""
    return getattr(strategy_cls, 'freq', 'daily')


def discover_params(strategy_cls):
    """Auto-discover tunable parameters from class annotations."""
    params = []

    # Try @param first
    try:
        discovered = strategy_cls.get_tunable_params()
        if discovered:
            return [(p.name, p.low, p.high, p.step) for p in discovered]
    except (AttributeError, Exception):
        pass

    # Fallback: annotation-based discovery
    try:
        discovered = strategy_cls.get_annotation_params()
        if discovered:
            return [(p.name, p.low, p.high, p.step) for p in discovered]
    except (AttributeError, Exception):
        pass

    # Manual fallback: inspect class annotations
    annotations = {}
    for cls in reversed(strategy_cls.__mro__):
        annotations.update(getattr(cls, '__annotations__', {}))

    for name, type_hint in annotations.items():
        if name.startswith('_') or name in ('name', 'warmup', 'freq'):
            continue
        default = getattr(strategy_cls, name, None)
        if default is None:
            continue

        if type_hint == int or (isinstance(default, int) and not isinstance(default, bool)):
            low = max(1, int(default * 0.3))
            high = int(default * 3.0)
            step = max(1, (high - low) // 20)
            params.append((name, low, high, step))
        elif type_hint == float or isinstance(default, float):
            low = max(0.01, default * 0.3)
            high = default * 3.0
            step = round((high - low) / 20, 4)
            params.append((name, low, high, step))

    return params


def optimize_single(version: str, n_trials: int = 30, phase: str = "coarse",
                    n_jobs: int = 1, seed: int = 42, probe_trials: int = 5):
    """Optimize a single strategy.

    Optimization flow:
    1. Probe phase: run `probe_trials` trials first
    2. If all probe trials return -999 (strategy errors), skip remaining trials
    3. Otherwise, run remaining trials to complete `n_trials` total

    Returns dict with results or None if failed.
    """
    from alphaforge.optimizer.base import StrategyOptimizer
    import time

    try:
        strategy_cls = load_strategy_class(version)
        freq = get_strategy_freq(strategy_cls)

        print(f"\n{'='*60}")
        print(f"Optimizing {version} | freq={freq} | phase={phase} | trials={n_trials}")
        print(f"{'='*60}")

        t0 = time.time()

        # Create optimizer
        opt = StrategyOptimizer(
            strategy_file=str(AG_DIR / f"{version}.py"),
            symbols=[SYMBOL],
            freq=freq,
            end=TRAIN_END,
            data_dir=get_data_dir(),
            initial_capital=DEFAULT_CAPITAL,
        )

        # Discover parameters
        params = discover_params(strategy_cls)
        if not params:
            print(f"  WARNING: No tunable parameters found for {version}, skipping")
            return None

        # Setup parameters based on phase
        if phase == "fine":
            coarse = load_coarse_results()
            if version in coarse and coarse[version].get("best_params"):
                best = coarse[version]["best_params"]
                for name, low, high, step in params:
                    if name in best:
                        val = best[name]
                        range_size = (high - low) * 0.3
                        new_low = max(low, val - range_size / 2)
                        new_high = min(high, val + range_size / 2)
                        new_step = step / 2 if step else None
                        opt.add_param(name, new_low, new_high, step=new_step)
                    else:
                        opt.add_param(name, low, high, step=step)
                print(f"  Fine-tuning around coarse best: {best}")
            else:
                print(f"  No coarse results for {version}, using full range")
                for name, low, high, step in params:
                    opt.add_param(name, low, high, step=step)
        else:
            for name, low, high, step in params:
                opt.add_param(name, low, high, step=step)

        print(f"  Parameters: {[(p[0], p[1], p[2]) for p in params]}")

        # --- PROBE PHASE: quick check if strategy works at all ---
        probe_n = min(probe_trials, n_trials)
        probe_result = opt.optimize(
            n_trials=probe_n,
            objective="sharpe",
            n_jobs=n_jobs,
            seed=seed,
        )

        # Check if all probe trials failed (-999 means strategy error)
        probe_values = probe_result.all_trials["value"].dropna().tolist()
        all_failed = all(v <= -900 for v in probe_values) if probe_values else True

        if all_failed:
            elapsed = time.time() - t0
            print(f"  EARLY STOP: All {probe_n} probe trials failed (-999). "
                  f"Strategy has code errors. Skipping. ({elapsed:.1f}s)")
            return {
                "version": version, "phase": phase, "freq": freq,
                "best_params": {}, "best_sharpe": -999.0,
                "n_trials": probe_n, "n_completed": probe_n,
                "early_stopped": True, "reason": "all_probe_trials_failed",
            }

        # --- MAIN PHASE: run remaining trials ---
        remaining = n_trials - probe_n
        if remaining > 0:
            result = opt.optimize(
                n_trials=remaining,
                objective="sharpe",
                n_jobs=n_jobs,
                seed=seed + 1000,
            )
        else:
            result = probe_result

        elapsed = time.time() - t0

        output = {
            "version": version,
            "phase": phase,
            "freq": freq,
            "best_params": result.best_params,
            "best_sharpe": float(result.best_value),
            "n_trials": n_trials,
            "n_completed": len(result.all_trials) + probe_n,
            "elapsed_seconds": round(elapsed, 1),
        }

        print(f"  Best Sharpe: {result.best_value:.4f}")
        print(f"  Best params: {result.best_params}")
        print(f"  Time: {elapsed:.1f}s")

        if phase == "coarse" and result.best_value < 0:
            print(f"  NOTE: Negative Sharpe, but keeping (may hedge in portfolio)")

        return output

    except Exception as e:
        print(f"  ERROR optimizing {version}: {e}")
        traceback.print_exc()
        return {"version": version, "phase": phase, "error": str(e)}


def load_coarse_results():
    """Load coarse optimization results."""
    if COARSE_RESULTS_FILE.exists():
        with open(COARSE_RESULTS_FILE) as f:
            data = json.load(f)
        return {r["version"]: r for r in data if "error" not in r}
    return {}


def load_results():
    """Load existing optimization results."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results, filepath):
    """Save optimization results (append mode — merges with existing)."""
    existing = []
    if filepath.exists():
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, Exception):
            existing = []

    # Merge: new results override existing for same version
    existing_map = {r["version"]: r for r in existing}
    for r in results:
        existing_map[r["version"]] = r

    merged = sorted(existing_map.values(), key=lambda r: int(r["version"][1:]))
    with open(filepath, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nResults saved to {filepath} ({len(merged)} total, {len(results)} new)")


def parse_strategy_range(s: str) -> list[str]:
    """Parse strategy range like 'v1-v50' or 'all' or 'v1,v5,v10'."""
    if s == "all":
        files = sorted(AG_DIR.glob("v*.py"))
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
    parser = argparse.ArgumentParser(description="All-Time AG Strategy Optimizer")
    parser.add_argument("--strategy", required=True, help="v1, v1-v50, all")
    parser.add_argument("--trials", type=int, default=50, help="Trials per strategy")
    parser.add_argument("--phase", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs per strategy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    versions = parse_strategy_range(args.strategy)
    print(f"Optimizing {len(versions)} strategies: {versions[0]}...{versions[-1]}")
    print(f"Phase: {args.phase} | Trials: {args.trials} | Symbol: {SYMBOL}")
    print(f"Training period: start ~ {TRAIN_END}")

    # Load already-done results to skip them
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
        filepath = AG_DIR / f"{version}.py"
        if not filepath.exists():
            print(f"  Skipping {version}: file not found")
            continue
        if version in already_done:
            print(f"  Skipping {version}: already optimized")
            continue

        r = optimize_single(version, n_trials=args.trials, phase=args.phase,
                           n_jobs=args.jobs, seed=args.seed)
        if r:
            results.append(r)

    # Save results
    if args.phase == "coarse":
        save_results(results, COARSE_RESULTS_FILE)
    else:
        save_results(results, RESULTS_FILE)

    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    positive = [r for r in successful if r.get("best_sharpe", 0) > 0]
    negative = [r for r in successful if r.get("best_sharpe", 0) <= 0]

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SUMMARY ({args.phase})")
    print(f"{'='*60}")
    print(f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}")
    print(f"Positive Sharpe: {len(positive)} | Negative Sharpe: {len(negative)}")
    if successful:
        sharpes = [r["best_sharpe"] for r in successful]
        print(f"Sharpe range: [{min(sharpes):.4f}, {max(sharpes):.4f}]")
        print(f"Sharpe mean: {np.mean(sharpes):.4f} | median: {np.median(sharpes):.4f}")


if __name__ == "__main__":
    main()
