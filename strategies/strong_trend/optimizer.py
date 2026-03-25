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
    # --- v21-v30 ---
    "v21": {
        "st_period": ("int", 7, 20), "st_mult": ("float", 2.0, 5.0),
        "adx_period": ("int", 10, 25), "adx_threshold": ("float", 18.0, 35.0),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v22": {
        "hma_period": ("int", 10, 40), "aroon_period": ("int", 15, 40),
        "cmf_period": ("int", 10, 30), "atr_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v23": {
        "kc_ema": ("int", 10, 30), "kc_mult": ("float", 1.0, 3.0),
        "roc_period": ("int", 10, 40), "roc_threshold": ("float", 2.0, 15.0),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v24": {
        "psar_af_step": ("float", 0.01, 0.04), "psar_af_max": ("float", 0.1, 0.3),
        "cci_period": ("int", 10, 30), "oi_period": ("int", 10, 30),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v25": {
        "slope_period": ("int", 10, 40), "vortex_period": ("int", 10, 25),
        "vol_period": ("int", 10, 30), "vol_threshold": ("float", 1.5, 3.5),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v26": {
        "squeeze_bb": ("int", 15, 30), "squeeze_kc": ("int", 15, 30),
        "adx_period": ("int", 10, 25), "fi_period": ("int", 8, 20),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v27": {
        "tenkan": ("int", 5, 15), "kijun": ("int", 15, 40),
        "macd_fast": ("int", 8, 20), "macd_slow": ("int", 20, 40),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v28": {
        "don_period": ("int", 25, 60), "chop_period": ("int", 10, 25),
        "chop_threshold": ("float", 38.0, 62.0), "ad_lookback": ("int", 5, 20),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v29": {
        "zlema_period": ("int", 10, 40), "rsi_period": ("int", 10, 25),
        "klinger_fast": ("int", 20, 45), "klinger_slow": ("int", 40, 70),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v30": {
        "st_period": ("int", 7, 20), "st_mult": ("float", 2.0, 5.0),
        "cop_wma": ("int", 5, 15), "cop_roc_long": ("int", 10, 20),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    # --- v31-v40 ---
    "v31": {
        "tema_period": ("int", 10, 40), "fisher_period": ("int", 5, 20),
        "obv_lookback": ("int", 5, 20), "atr_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v32": {
        "ribbon_base": ("int", 5, 15), "tsi_long": ("int", 15, 35),
        "tsi_short": ("int", 8, 20), "vroc_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v33": {
        "t3_period": ("int", 3, 10), "t3_vfactor": ("float", 0.5, 0.9),
        "aroon_period": ("int", 15, 40), "climax_period": ("int", 10, 30),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v34": {
        "mcg_period": ("int", 8, 25), "ppo_fast": ("int", 8, 20),
        "ppo_slow": ("int", 20, 40), "oi_period": ("int", 10, 30),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v35": {
        "alma_period": ("int", 5, 20), "alma_offset": ("float", 0.7, 0.95),
        "stochrsi_period": ("int", 10, 25), "emv_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v36": {
        "dema_period": ("int", 10, 40), "uo_p1": ("int", 5, 12),
        "uo_p2": ("int", 10, 20), "uo_p3": ("int", 20, 40),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v37": {
        "fractal_period": ("int", 2, 5), "rwi_period": ("int", 10, 25),
        "fi_period": ("int", 8, 20), "atr_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v38": {
        "hma_period": ("int", 10, 40), "ergo_short": ("int", 3, 10),
        "ergo_long": ("int", 15, 30), "ad_lookback": ("int", 5, 20),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v39": {
        "kc_ema": ("int", 10, 30), "kc_mult": ("float", 1.0, 3.0),
        "cmo_period": ("int", 10, 25), "vol_threshold": ("float", 1.5, 3.5),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v40": {
        "psar_af_step": ("float", 0.01, 0.04), "psar_af_max": ("float", 0.1, 0.3),
        "kst_signal": ("int", 5, 15), "klinger_fast": ("int", 20, 45),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    # --- v41-v45 (multi-timeframe) ---
    "v41": {
        "st_period": ("int", 7, 20), "st_mult": ("float", 2.0, 5.0),
        "rsi_period": ("int", 10, 25), "rsi_entry": ("float", 25.0, 45.0),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v42": {
        "adx_period": ("int", 10, 25), "adx_threshold": ("float", 18.0, 35.0),
        "ema_fast": ("int", 10, 30), "ema_slow": ("int", 40, 80),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v43": {
        "tenkan": ("int", 5, 15), "kijun": ("int", 15, 40),
        "stoch_k": ("int", 10, 25), "stoch_d": ("int", 2, 5),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v44": {
        "don_period": ("int", 25, 60), "roc_period": ("int", 10, 40),
        "roc_threshold": ("float", 1.0, 8.0), "atr_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v45": {
        "vortex_period": ("int", 10, 25), "cci_period": ("int", 10, 30),
        "cci_entry": ("float", -150.0, -50.0), "atr_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    # --- v46-v48 (minimal) ---
    "v46": {
        "st_period": ("int", 7, 20), "st_mult": ("float", 2.0, 5.0),
        "atr_period": ("int", 10, 25), "atr_trail_mult": ("float", 3.5, 6.0),
        "new_high_lookback": ("int", 10, 40),
    },
    "v47": {
        "adx_period": ("int", 10, 25), "adx_threshold": ("float", 18.0, 35.0),
        "chand_period": ("int", 15, 35), "chand_mult": ("float", 2.0, 5.0),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v48": {
        "rsq_period": ("int", 20, 60), "don_period": ("int", 25, 60),
        "atr_period": ("int", 10, 25), "atr_trail_mult": ("float", 3.0, 5.5),
        "rsq_threshold": ("float", 0.3, 0.7),
    },
    # --- v49-v50 (regime) ---
    "v49": {
        "hurst_lag": ("int", 10, 30), "st_period": ("int", 7, 20),
        "st_mult": ("float", 2.0, 5.0), "vol_threshold": ("float", 1.5, 3.5),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
    "v50": {
        "yz_period": ("int", 10, 30), "don_period": ("int", 25, 60),
        "obv_lookback": ("int", 5, 20), "atr_period": ("int", 10, 25),
        "atr_trail_mult": ("float", 3.0, 5.5),
    },
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


def create_strategy_with_params(version: str, params: dict):
    """Create a strategy instance with given parameters."""
    cls = load_strategy_class(version)
    instance = cls()
    for k, v in params.items():
        if hasattr(instance, k):
            setattr(instance, k, v)
    return instance


def _map_freq(freq: str) -> tuple:
    """Map strategy freq to (AlphaForge load freq, resample factor).

    AlphaForge native: 1min, 5min, 10min, 15min, 30min, 60min, daily.
    Non-native frequencies are loaded at a base freq and resampled.

    Returns (load_freq, resample_factor) where factor=1 means no resample.
    """
    NATIVE = {"1min", "5min", "10min", "15min", "30min", "60min", "daily"}
    if freq in NATIVE:
        return freq, 1

    # Map non-native freqs → (base_freq, how_many_bars_to_merge)
    resample_map = {
        "1h":   ("60min", 1),   # 60min IS 1h
        "20min": ("10min", 2),  # 2 × 10min = 20min
        "4h":   ("60min", 4),   # 4 × 60min = 4h
    }
    if freq in resample_map:
        return resample_map[freq]

    # Fallback: try to use as-is
    return freq, 1


def _resample_bars(bars, step: int):
    """Resample BarArray by grouping every `step` bars (e.g. step=4 for 60min→4h)."""
    from alphaforge.data.bardata import BarArray
    n = len(bars)
    indices = list(range(step - 1, n, step))
    if not indices:
        return bars

    new_len = len(indices)

    def _ohlcv_resample(src, mode):
        arr = np.array(src, dtype=np.float64)
        out = np.empty(new_len, dtype=np.float64)
        for j, end_idx in enumerate(indices):
            s = max(0, end_idx - step + 1)
            chunk = arr[s:end_idx + 1]
            if mode == 'first':
                out[j] = chunk[0]
            elif mode == 'max':
                out[j] = chunk.max()
            elif mode == 'min':
                out[j] = chunk.min()
            elif mode == 'last':
                out[j] = chunk[-1]
            elif mode == 'sum':
                out[j] = chunk.sum()
        return out

    def _pick(src):
        return np.array(src)[indices]

    return BarArray(
        datetime_arr=_pick(bars.datetime),
        open_arr=_ohlcv_resample(bars.open, 'first'),
        high_arr=_ohlcv_resample(bars.high, 'max'),
        low_arr=_ohlcv_resample(bars.low, 'min'),
        close_arr=_ohlcv_resample(bars.close, 'last'),
        volume_arr=_ohlcv_resample(bars.volume, 'sum'),
        amount_arr=_ohlcv_resample(bars.amount, 'sum'),
        oi_arr=_ohlcv_resample(bars.oi, 'last'),
        trading_day_arr=_pick(bars.trading_day),
        open_raw_arr=_ohlcv_resample(bars.open_raw, 'first'),
        high_raw_arr=_ohlcv_resample(bars.high_raw, 'max'),
        low_raw_arr=_ohlcv_resample(bars.low_raw, 'min'),
        close_raw_arr=_ohlcv_resample(bars.close_raw, 'last'),
        origin_symbol_arr=_pick(bars.origin_symbol),
        factor_arr=_pick(bars.factor),
        is_rollover_arr=_pick(bars.is_rollover),
    )


def run_single_backtest(strategy, symbol, start, end, freq="daily"):
    """Run a single backtest, return Sharpe (or -999 on failure)."""
    try:
        loader = MarketDataLoader(DATA_DIR)
        spec_manager = ContractSpecManager()

        load_freq, resample_factor = _map_freq(freq)
        bars = loader.load(symbol, freq=load_freq, start=start, end=end)
        if bars is None or len(bars) < strategy.warmup + 20:
            return -999.0

        # Resample if needed (e.g. 60min → 4h = factor 4)
        if resample_factor > 1:
            bars = _resample_bars(bars, resample_factor)

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
