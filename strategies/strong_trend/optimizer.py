"""
Optuna Optimizer for Strong Trend Strategies
=============================================
Optimizes each strategy's 5 parameters on training set commodities.
Goal: maximize average Sharpe across multiple symbols to avoid overfitting.

Usage:
    python strategies/strong_trend/optimizer.py --strategy v1 --trials 150
    python strategies/strong_trend/optimizer.py --strategy all --trials 100
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

import numpy as np
import optuna
from optuna.samplers import TPESampler

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester

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

# Parameter search spaces for each strategy version
PARAM_SPACES = {
    "v1": {
        "st_period":      ("int", 7, 20),
        "st_mult":        ("float", 2.0, 5.0),
        "roc_period":     ("int", 10, 40),
        "roc_threshold":  ("float", 2.0, 15.0),
        "vol_threshold":  ("float", 1.5, 3.5),
    },
    "v2": {
        "kama_period":    ("int", 5, 20),
        "rsq_period":     ("int", 20, 60),
        "rsq_threshold":  ("float", 0.3, 0.7),
        "atr_period":     ("int", 10, 25),
        "trail_mult":     ("float", 2.5, 5.0),
    },
    "v3": {
        "don_period":     ("int", 30, 80),
        "adx_period":     ("int", 10, 25),
        "adx_threshold":  ("float", 18.0, 35.0),
        "chand_period":   ("int", 15, 35),
        "chand_mult":     ("float", 2.0, 5.0),
    },
    "v4": {
        "ema_fast":       ("int", 10, 30),
        "ema_slow":       ("int", 40, 100),
        "macd_fast":      ("int", 8, 20),
        "macd_slow":      ("int", 20, 40),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v5": {
        "hma_period":     ("int", 10, 40),
        "cci_period":     ("int", 10, 30),
        "cci_threshold":  ("float", 50.0, 200.0),
        "climax_period":  ("int", 10, 30),
        "atr_trail_mult": ("float", 2.5, 5.0),
    },
    "v6": {
        "tenkan":         ("int", 5, 15),
        "kijun":          ("int", 15, 40),
        "rsi_period":     ("int", 10, 25),
        "rsi_threshold":  ("float", 40.0, 60.0),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v7": {
        "psar_af_step":   ("float", 0.01, 0.04),
        "psar_af_max":    ("float", 0.1, 0.3),
        "tsi_long":       ("int", 15, 35),
        "tsi_short":      ("int", 8, 20),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v8": {
        "slope_period":   ("int", 10, 40),
        "chop_period":    ("int", 10, 25),
        "chop_threshold": ("float", 38.0, 62.0),
        "natr_period":    ("int", 10, 25),
        "atr_trail_mult": ("float", 2.5, 5.0),
    },
    "v9": {
        "st_period":      ("int", 7, 20),
        "st_mult":        ("float", 2.0, 5.0),
        "stochrsi_period": ("int", 10, 25),
        "fi_period":      ("int", 8, 20),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v10": {
        "tema_period":    ("int", 10, 40),
        "adx_period":     ("int", 10, 25),
        "adx_threshold":  ("float", 18.0, 35.0),
        "bb_period":      ("int", 15, 30),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v11": {
        "vortex_period":  ("int", 10, 25),
        "roc_period":     ("int", 10, 40),
        "roc_threshold":  ("float", 2.0, 15.0),
        "oi_period":      ("int", 10, 30),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v12": {
        "aroon_period":   ("int", 15, 40),
        "ppo_fast":       ("int", 8, 20),
        "ppo_slow":       ("int", 20, 40),
        "vol_mom_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v13": {
        "zlema_period":   ("int", 10, 40),
        "fisher_period":  ("int", 5, 20),
        "re_period":      ("int", 10, 25),
        "re_threshold":   ("float", 1.0, 2.0),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v14": {
        "don_period":     ("int", 25, 60),
        "rwi_period":     ("int", 10, 25),
        "klinger_fast":   ("int", 20, 45),
        "klinger_slow":   ("int", 40, 70),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v15": {
        "ribbon_base":    ("int", 5, 15),
        "cmo_period":     ("int", 10, 25),
        "cmo_threshold":  ("float", 10.0, 40.0),
        "atr_period":     ("int", 10, 25),
        "atr_trail_mult": ("float", 2.5, 5.0),
    },
    "v16": {
        "t3_period":      ("int", 3, 10),
        "t3_vfactor":     ("float", 0.5, 0.9),
        "ergo_short":     ("int", 3, 10),
        "ergo_long":      ("int", 15, 30),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v17": {
        "alma_period":    ("int", 5, 20),
        "alma_offset":    ("float", 0.7, 0.95),
        "uo_p1":          ("int", 5, 12),
        "uo_p2":          ("int", 10, 20),
        "atr_trail_mult": ("float", 2.5, 5.0),
    },
    "v18": {
        "dema_period":    ("int", 10, 40),
        "kst_signal":     ("int", 5, 15),
        "nr7_lookback":   ("int", 5, 10),
        "atr_period":     ("int", 10, 25),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v19": {
        "mcg_period":     ("int", 8, 25),
        "cop_wma":        ("int", 5, 15),
        "cop_roc_long":   ("int", 10, 20),
        "cop_roc_short":  ("int", 8, 15),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
    "v20": {
        "fractal_period": ("int", 2, 5),
        "mass_ema":       ("int", 5, 15),
        "mass_sum":       ("int", 15, 35),
        "vroc_period":    ("int", 10, 25),
        "atr_trail_mult": ("float", 2.0, 5.0),
    },
}

# Strategy class names mapping
STRATEGY_CLASSES = {
    "v1": "StrongTrendV1",
    "v2": "StrongTrendV2",
    "v3": "DonchianADXChandelierStrategy",
    "v4": "StrongTrendV4",
    "v5": "StrongTrendV5",
    "v6": "StrongTrendV6",
    "v7": "StrongTrendV7",
    "v8": "StrongTrendV8",
    "v9": "StrongTrendV9",
    "v10": "StrongTrendV10",
    "v11": "StrongTrendV11",
    "v12": "StrongTrendV12",
    "v13": "StrongTrendV13",
    "v14": "StrongTrendV14",
    "v15": "StrongTrendV15",
    "v16": "StrongTrendV16",
    "v17": "StrongTrendV17",
    "v18": "StrongTrendV18",
    "v19": "StrongTrendV19",
    "v20": "StrongTrendV20",
}


# =========================================================================
# Core functions
# =========================================================================

def load_strategy_class(version: str):
    """Dynamically import and return the strategy class."""
    mod = importlib.import_module(f"strategies.strong_trend.{version}")
    class_name = STRATEGY_CLASSES[version]
    return getattr(mod, class_name)


def create_strategy_with_params(version: str, params: dict):
    """Create a strategy instance with given parameters."""
    cls = load_strategy_class(version)
    instance = cls()
    for k, v in params.items():
        if hasattr(instance, k):
            setattr(instance, k, v)
    return instance


def run_single_backtest(strategy, symbol, start, end, freq="daily"):
    """Run a single backtest, return Sharpe (or -999 on failure)."""
    try:
        loader = MarketDataLoader(DATA_DIR)
        spec_manager = ContractSpecManager()

        bars = loader.load(symbol, freq=freq, start=start, end=end)
        if bars is None or len(bars) < strategy.warmup + 20:
            return -999.0

        engine = EventDrivenBacktester(
            spec_manager=spec_manager,
            initial_capital=1_000_000,
            slippage_ticks=1.0,
        )
        result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)
        sharpe = result.sharpe
        if np.isnan(sharpe) or np.isinf(sharpe):
            return -999.0
        return float(sharpe)
    except Exception:
        return -999.0


def evaluate_strategy(version: str, params: dict) -> float:
    """Evaluate a strategy across all training symbols. Returns mean Sharpe."""
    strategy = create_strategy_with_params(version, params)
    freq = strategy.freq

    sharpes = []
    for symbol in TRAINING_SYMBOLS:
        start, end = TRAINING_PERIODS[symbol]
        s = run_single_backtest(strategy, symbol, start, end, freq=freq)
        if s > -900:  # Valid result
            sharpes.append(s)

    if len(sharpes) < 3:  # Need at least 3 valid results
        return -10.0
    return float(np.mean(sharpes))


def suggest_params(trial: optuna.Trial, version: str) -> dict:
    """Use Optuna trial to suggest parameters for a strategy version."""
    space = PARAM_SPACES[version]
    params = {}
    for name, (dtype, low, high) in space.items():
        if dtype == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif dtype == "float":
            params[name] = trial.suggest_float(name, low, high)
    return params


def optimize_strategy(version: str, n_trials: int = 150, verbose: bool = True):
    """Run Optuna optimization for a single strategy version."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing {version} ({STRATEGY_CLASSES[version]})")
        print(f"Trials: {n_trials}, Training symbols: {TRAINING_SYMBOLS}")
        print(f"{'='*60}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"strong_trend_{version}",
    )

    def objective(trial):
        params = suggest_params(trial, version)
        return evaluate_strategy(version, params)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best = study.best_trial
    if verbose:
        print(f"\nBest trial #{best.number}:")
        print(f"  Mean Sharpe: {best.value:.4f}")
        print(f"  Params: {best.params}")

    return {
        "version": version,
        "best_sharpe": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
        "n_completed": len(study.trials),
    }


def save_results(results: list, output_path: str = None):
    """Save optimization results to JSON."""
    if output_path is None:
        output_path = os.path.join(
            QBASE_ROOT, "strategies", "strong_trend", "optimization_results.json"
        )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize strong trend strategies")
    parser.add_argument(
        "--strategy", default="v1",
        help="Strategy version (v1-v20) or 'all' for all strategies"
    )
    parser.add_argument("--trials", type=int, default=150, help="Optuna trials per strategy")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.strategy == "all":
        versions = [f"v{i}" for i in range(1, 21)]
    else:
        versions = [args.strategy]

    all_results = []
    for ver in versions:
        result = optimize_strategy(ver, n_trials=args.trials, verbose=verbose)
        all_results.append(result)
        print(f"  {ver}: best Sharpe = {result['best_sharpe']:.4f}")

    save_results(all_results)

    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x["best_sharpe"], reverse=True):
        print(f"  {r['version']:>4s}: Sharpe = {r['best_sharpe']:>8.4f}  params = {r['best_params']}")
