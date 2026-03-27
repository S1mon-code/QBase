"""
Medium Trend 策略优化器

品种无关策略，在多品种训练集上优化，目标 = mean(Sharpe across training symbols)。
只做多，捕捉涨幅 20%-80%、持续 2-4 个月的中等趋势。

训练集：35 段中趋势行情（见 trend/MEDIUM_TRENDS.md）
测试集（Simon 指定，不参与优化）：
  - PG 2020-03-30 ~ 2020-07-27
  - UR 2021-05-27 ~ 2021-10-12
  - SC 2020-11-02 ~ 2021-02-24
  - SN 2025-09-29 ~ 2026-01-24
  - JM 2023-05-31 ~ 2023-09-16

用法:
    python optimizer.py --strategy v1 --trials 30 --phase coarse
    python optimizer.py --strategy all --trials 30 --phase coarse
    python optimizer.py --strategy v1 --trials 50 --phase fine
"""

import sys
import os
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

logging.getLogger("alphaforge").setLevel(logging.ERROR)
logging.getLogger("alphaforge.engine").setLevel(logging.ERROR)

MT_DIR = Path(__file__).resolve().parent
COARSE_RESULTS_FILE = MT_DIR / "optimization_coarse.json"
RESULTS_FILE = MT_DIR / "optimization_results.json"

# ── Training segments: symbol, start, end ──
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

# Frequency-adaptive segment selection
# Higher frequency = fewer segments (for speed), lower frequency = all segments (for robustness)
# Subset covers diverse sectors: black(J,I), energy(FU,SC), nonferrous(AG,NI), agri(CF,RU), building(SA,FG)
OPTIM_SEGMENTS_SMALL = [  # 12 segments for 30min/1h (diverse but fast)
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

def get_optim_segments(freq: str):
    """Select training segments based on strategy frequency."""
    if freq in ("daily", "4h"):
        return TRAIN_SEGMENTS  # 35 segments
    elif freq in ("1h", "30min"):
        return OPTIM_SEGMENTS_SMALL  # 12 segments
    else:  # 5min, 10min — should be skipped but just in case
        return OPTIM_SEGMENTS_SMALL[:8]  # 8 segments

DEFAULT_CAPITAL = 1_000_000


def load_strategy_class(version: str):
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


def discover_params(strategy_cls):
    params = []
    try:
        discovered = strategy_cls.get_tunable_params()
        if discovered:
            return [(p.name, p.low, p.high, p.step) for p in discovered]
    except (AttributeError, Exception):
        pass
    try:
        discovered = strategy_cls.get_annotation_params()
        if discovered:
            return [(p.name, p.low, p.high, p.step) for p in discovered]
    except (AttributeError, Exception):
        pass
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
                    seed: int = 42, probe_trials: int = 5):
    from alphaforge.data.market import MarketDataLoader
    from alphaforge.data.contract_specs import ContractSpecManager
    from alphaforge.engine.event_driven import EventDrivenBacktester

    try:
        strategy_cls = load_strategy_class(version)
        freq = get_strategy_freq(strategy_cls)
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"Optimizing {version} | freq={freq} | phase={phase} | trials={n_trials}")
        print(f"{'='*60}")

        params = discover_params(strategy_cls)
        if not params:
            print(f"  WARNING: No tunable parameters found for {version}, skipping")
            return None

        # Load coarse results for fine phase
        if phase == "fine":
            coarse = load_coarse_results()
            if version in coarse and coarse[version].get("best_params"):
                best = coarse[version]["best_params"]
                narrowed = []
                for name, low, high, step in params:
                    if name in best:
                        val = best[name]
                        rng = (high - low) * 0.3
                        narrowed.append((name, max(low, val - rng/2), min(high, val + rng/2), step/2 if step else None))
                    else:
                        narrowed.append((name, low, high, step))
                params = narrowed
                print(f"  Fine-tuning around coarse best: {best}")

        print(f"  Parameters: {[(p[0], round(p[1],2), round(p[2],2)) for p in params]}")
        segments = get_optim_segments(freq)
        print(f"  Training on {len(segments)} segments (freq={freq})")

        # Load data for all training segments
        loader = MarketDataLoader(get_data_dir())
        spec_mgr = ContractSpecManager()

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna required: pip install optuna")

        def objective(trial):
            # Sample parameters
            trial_params = {}
            for name, low, high, step in params:
                if isinstance(getattr(strategy_cls, name, 0.0), int):
                    trial_params[name] = trial.suggest_int(name, int(low), int(high),
                                                           step=int(step) if step else 1)
                else:
                    trial_params[name] = trial.suggest_float(name, low, high, step=step)

            sharpes = []
            for sym, start, end in segments:
                try:
                    bars = loader.load(sym, freq=freq, start=start, end=end)
                    strat = strategy_cls()
                    for k, v in trial_params.items():
                        setattr(strat, k, v)
                    engine = EventDrivenBacktester(
                        spec_manager=spec_mgr, initial_capital=DEFAULT_CAPITAL)
                    result = engine.run(strat, {sym: bars})
                    s = result.sharpe if result.sharpe is not None and not np.isnan(result.sharpe) else -1.0
                    sharpes.append(s)
                except Exception:
                    sharpes.append(-1.0)

            if not sharpes:
                return -999.0
            # Objective: mean + consistency bonus
            mean_s = np.mean(sharpes)
            min_s = min(sharpes)
            return mean_s + 0.3 * min_s

        # Probe phase
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=min(probe_trials, n_trials), show_progress_bar=False)

        probe_values = [t.value for t in study.trials if t.value is not None]
        if all(v <= -900 for v in probe_values):
            elapsed = time.time() - t0
            print(f"  EARLY STOP: All probe trials failed. ({elapsed:.1f}s)")
            return {"version": version, "phase": phase, "freq": freq,
                    "best_params": {}, "best_sharpe": -999.0,
                    "early_stopped": True, "elapsed_seconds": round(elapsed, 1)}

        # Main phase
        remaining = n_trials - len(study.trials)
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, show_progress_bar=False)

        elapsed = time.time() - t0
        best = study.best_trial
        output = {
            "version": version, "phase": phase, "freq": freq,
            "best_params": best.params,
            "best_sharpe": float(best.value),
            "n_trials": n_trials,
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"  Best Sharpe (mean+bonus): {best.value:.4f}")
        print(f"  Best params: {best.params}")
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


def parse_strategy_range(s: str) -> list:
    if s == "all":
        return [f.stem for f in sorted(MT_DIR.glob("v*.py"))]
    if "-" in s and s.count("-") == 1:
        parts = s.split("-")
        return [f"v{i}" for i in range(int(parts[0].replace("v","")), int(parts[1].replace("v",""))+1)]
    if "," in s:
        return [v.strip() for v in s.split(",")]
    return [s]


def main():
    parser = argparse.ArgumentParser(description="Medium Trend Strategy Optimizer")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--phase", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--seed", type=int, default=42)
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
    print(f"Training: 35 segments (daily/4h) / 12 segments (30min/1h) | Trials: {args.trials}")

    results = []
    for version in versions:
        if not (MT_DIR / f"{version}.py").exists():
            continue
        if version in already_done:
            print(f"  Skipping {version}: already optimized")
            continue
        r = optimize_single(version, n_trials=args.trials, phase=args.phase, seed=args.seed)
        if r:
            results.append(r)

    save_results(results, result_file)

    successful = [r for r in results if "error" not in r and r.get("best_sharpe", -999) > -900]
    print(f"\nSUMMARY: {len(results)} processed | {len(successful)} valid | "
          f"{sum(1 for r in successful if r['best_sharpe']>0)} positive Sharpe")
    if successful:
        sharpes = [r["best_sharpe"] for r in successful]
        print(f"Sharpe: [{min(sharpes):.3f}, {max(sharpes):.3f}] mean={np.mean(sharpes):.3f}")


if __name__ == "__main__":
    main()
