"""
Medium Trend 策略优化器（optimizer_core 版本）

品种无关策略，在多品种多时段训练集上优化。
只做多，捕捉涨幅 20%-80%、持续 2-4 个月的中等趋势。

使用 optimizer_core 的加权多维评分系统 (0-10):
  score = 0.60×S_sharpe + 0.15×S_risk + 0.15×S_quality + 0.10×S_stability

训练集：35 段中趋势行情（见 trend/MEDIUM_TRENDS.md）
测试集（不参与优化）：PG, UR, SC, SN, JM

用法:
    python optimizer.py --strategy v1 --trials 50 --phase coarse
    python optimizer.py --strategy all --trials 50 --phase coarse
    python optimizer.py --strategy v1 --trials 100 --phase fine
    python optimizer.py --strategy v1 --trials 50 --multi-seed
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
    narrow_param_space, map_freq, resample_bars,
)

logging.getLogger("alphaforge").setLevel(logging.ERROR)
logging.getLogger("alphaforge.engine").setLevel(logging.ERROR)

MT_DIR = Path(__file__).resolve().parent
COARSE_RESULTS_FILE = MT_DIR / "optimization_coarse.json"
RESULTS_FILE = MT_DIR / "optimization_results.json"

DEFAULT_CAPITAL = 1_000_000

# ── Training segments: (symbol, start, end) ──
# All segments from MEDIUM_TRENDS.md EXCEPT the 5 test segments
TRAIN_SEGMENTS = [
    ("EC", "2024-09-19", "2025-02-10"),
    ("ZC", "2021-08-09", "2021-10-11"),
    ("JM", "2021-07-23", "2021-10-12"),
    ("JM", "2016-09-14", "2016-11-17"),
    ("B", "2017-01-04", "2017-03-22"),
    ("EC", "2023-09-25", "2023-12-26"),
    ("LC", "2025-11-13", "2026-01-20"),
    ("J", "2016-08-08", "2016-12-02"),
    ("I", "2016-09-20", "2016-12-07"),
    ("SM", "2016-05-27", "2016-10-13"),
    ("CF", "2010-08-24", "2010-11-05"),
    ("SA", "2023-08-19", "2023-12-12"),
    ("AG", "2025-09-02", "2025-12-24"),
    ("J", "2015-12-28", "2016-04-23"),
    ("SF", "2021-08-04", "2021-10-15"),
    ("NI", "2022-01-07", "2022-03-09"),
    ("J", "2017-05-31", "2017-09-13"),
    ("PS", "2025-06-23", "2025-09-15"),
    ("AG", "2020-04-17", "2020-08-11"),
    ("SC", "2021-12-18", "2022-03-09"),
    ("JM", "2025-06-03", "2025-07-25"),
    ("SM", "2021-05-26", "2021-10-11"),
    ("FU", "2016-09-12", "2016-12-14"),
    ("J", "2021-05-25", "2021-09-09"),
    ("HC", "2015-12-25", "2016-04-21"),
    ("LU", "2022-02-18", "2022-06-14"),
    ("I", "2015-12-25", "2016-04-21"),
    ("FG", "2021-01-14", "2021-05-13"),
    ("FU", "2021-12-18", "2022-03-09"),
    ("EB", "2020-12-29", "2021-04-24"),
    ("P", "2021-12-20", "2022-04-18"),
    ("I", "2019-03-11", "2019-07-03"),
    ("FU", "2015-08-26", "2016-01-11"),
    ("RU", "2010-09-16", "2011-02-09"),
    ("SP", "2020-11-05", "2021-02-27"),
]

# Test segments (not used in optimization)
TEST_SEGMENTS = [
    ("PG", "2020-03-30", "2020-07-27"),
    ("UR", "2021-05-27", "2021-10-12"),
    ("SC", "2020-11-02", "2021-02-24"),
    ("SN", "2025-09-29", "2026-01-24"),
    ("JM", "2023-05-31", "2023-09-16"),
]

# Frequency-adaptive segment selection (fewer segments for higher freq = faster)
OPTIM_SEGMENTS_SMALL = [  # 12 segments for 30min/1h
    ("J", "2016-08-08", "2016-12-02"),
    ("I", "2019-03-11", "2019-07-03"),
    ("AG", "2020-04-17", "2020-08-11"),
    ("NI", "2022-01-07", "2022-03-09"),
    ("FU", "2016-09-12", "2016-12-14"),
    ("SC", "2021-12-18", "2022-03-09"),
    ("SA", "2023-08-19", "2023-12-12"),
    ("CF", "2010-08-24", "2010-11-05"),
    ("FG", "2021-01-14", "2021-05-13"),
    ("EB", "2020-12-29", "2021-04-24"),
    ("RU", "2010-09-16", "2011-02-09"),
    ("LC", "2025-11-13", "2026-01-20"),
]


def get_optim_segments(freq):
    """Select training segments based on strategy frequency."""
    if freq in ("daily", "4h"):
        return TRAIN_SEGMENTS        # 35 segments
    elif freq in ("1h", "30min", "60min"):
        return OPTIM_SEGMENTS_SMALL  # 12 segments
    else:  # 5min, 10min
        return OPTIM_SEGMENTS_SMALL[:8]  # 8 segments


def load_strategy_class(version):
    filepath = MT_DIR / f"{version}.py"
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


def optimize_single(version, n_trials=50, phase="coarse",
                    seed=42, probe_trials=5, multi_seed=False):
    """Optimize a single strategy using optimizer_core multi-dimensional scoring."""
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

        print(f"  Parameters: {[(p['name'], round(p['low'],2), round(p['high'],2)) for p in param_specs]}")

        segments = get_optim_segments(freq)
        print(f"  Training on {len(segments)} segments (freq={freq})")

        t0 = time.time()
        data_dir = get_data_dir()

        def objective_fn(params):
            """Evaluate across multiple training segments using composite_objective."""
            strategy = create_strategy_with_params(strategy_cls, params)
            results = []
            for sym, start, end in segments:
                r = run_single_backtest(
                    strategy, sym, start, end,
                    freq=freq, data_dir=data_dir,
                    initial_capital=DEFAULT_CAPITAL,
                )
                if r['sharpe'] > -900:
                    results.append(r)

            if len(results) < max(3, len(segments) // 3):
                return -10.0

            return composite_objective(results, min_valid=3, freq=freq)

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
                study_name=f"mt_{version}",
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
                seed=seed,
                study_name=f"mt_{version}",
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

        print(f"  Best Score: {output['best_sharpe']:.4f}/10")
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
            return {r["version"]: r for r in json.load(f) if "error" not in r}
    return {}


def save_results(results, filepath):
    existing = []
    if filepath.exists():
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except Exception:
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
        return [f.stem for f in sorted(MT_DIR.glob("v*.py"))]
    if "-" in s and s.count("-") == 1:
        parts = s.split("-")
        return [f"v{i}" for i in range(int(parts[0].replace("v", "")), int(parts[1].replace("v", "")) + 1)]
    if "," in s:
        return [v.strip() for v in s.split(",")]
    return [s]


def main():
    parser = argparse.ArgumentParser(description="Medium Trend Strategy Optimizer")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--phase", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run multi-seed optimization (3 seeds) for robustness")
    args = parser.parse_args()

    versions = parse_strategy_range(args.strategy)
    result_file = COARSE_RESULTS_FILE if args.phase == "coarse" else RESULTS_FILE

    # Skip already done
    already_done = set()
    if result_file.exists():
        try:
            with open(result_file) as f:
                already_done = {r["version"] for r in json.load(f)}
        except Exception:
            pass

    print(f"Medium Trend Optimizer | {len(versions)} strategies | phase={args.phase}")
    print(f"Scoring: 0.60×Sharpe + 0.15×Risk + 0.15×Quality + 0.10×Stability")
    print(f"Training: 35 segments (daily/4h) / 12 segments (30min/1h) | Trials: {args.trials}")

    results = []
    for version in versions:
        if not (MT_DIR / f"{version}.py").exists():
            continue
        if version in already_done:
            print(f"  Skipping {version}: already optimized")
            continue
        r = optimize_single(version, n_trials=args.trials, phase=args.phase,
                           seed=args.seed, multi_seed=args.multi_seed)
        if r:
            results.append(r)

    save_results(results, result_file)

    successful = [r for r in results if "error" not in r and r.get("best_sharpe", -999) > -9]
    positive = [r for r in successful if r.get("best_sharpe", 0) > 0]
    robust = [r for r in positive if r.get("robustness", {}).get("is_robust")]

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SUMMARY ({args.phase})")
    print(f"{'='*60}")
    print(f"Total: {len(results)} | Valid: {len(successful)} | Positive: {len(positive)} | Robust: {len(robust)}")
    if successful:
        scores = [r["best_sharpe"] for r in successful]
        print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"Score mean: {np.mean(scores):.4f} | median: {np.median(scores):.4f}")


if __name__ == "__main__":
    main()
