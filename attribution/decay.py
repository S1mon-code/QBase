"""Alpha Decay Detection — rolling Information Coefficient analysis for indicators.

Usage:
    python -m attribution.decay \
        --indicator volume_momentum \
        --symbols AG,J,I,ZC \
        --window 252 \
        --horizon 5
"""
import sys
import argparse
import importlib
import warnings
from pathlib import Path
from datetime import datetime

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# =========================================================================
# Indicator name -> (module_path, function_name, required_arrays) mapping
# =========================================================================

# Maps indicator names to their import location and required input arrays.
# "required_arrays" defines which price arrays the indicator needs:
#   c=closes, v=volumes, h=highs, l=lows, o=opens
_INDICATOR_MAP = {
    # Volume
    "volume_momentum": ("indicators.volume.volume_momentum", "volume_momentum", "v"),
    "relative_volume": ("indicators.volume.volume_momentum", "relative_volume", "v"),
    "obv": ("indicators.volume.obv", "obv", "cv"),
    "mfi": ("indicators.volume.mfi", "mfi", "hlcv"),
    "cmf": ("indicators.volume.cmf", "cmf", "hlcv"),
    "vroc": ("indicators.volume.vroc", "vroc", "v"),
    "force_index": ("indicators.volume.force_index", "force_index", "cv"),
    # Momentum
    "rsi": ("indicators.momentum.rsi", "rsi", "c"),
    "roc": ("indicators.momentum.roc", "rate_of_change", "c"),
    "cci": ("indicators.momentum.cci", "cci", "hlc"),
    "cmo": ("indicators.momentum.cmo", "cmo", "c"),
    "ppo": ("indicators.momentum.ppo", "ppo", "c"),
    "tsmom": ("indicators.momentum.tsmom", "tsmom", "c"),
    "macd": ("indicators.momentum.macd", "macd", "c"),
    # Trend
    "adx": ("indicators.trend.adx", "adx", "hlc"),
    "aroon": ("indicators.trend.aroon", "aroon", "hl"),
    "supertrend": ("indicators.trend.supertrend", "supertrend", "hlc"),
    "trend_intensity": ("indicators.trend.trend_intensity", "trend_intensity", "c"),
}


def _load_indicator_func(indicator_name: str):
    """Load an indicator function by name.

    Returns:
        (callable, required_arrays_str) where required_arrays_str is like "c", "hlcv", etc.
    """
    if indicator_name in _INDICATOR_MAP:
        mod_path, func_name, arrays = _INDICATOR_MAP[indicator_name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, func_name), arrays

    # Fallback: try to find in indicator subdirectories
    for category in ("momentum", "trend", "volume", "volatility", "regime",
                     "structure", "spread", "ml", "microstructure", "seasonality"):
        try:
            mod = importlib.import_module(f"indicators.{category}.{indicator_name}")
            func = getattr(mod, indicator_name, None)
            if func is not None:
                return func, "c"  # Default to closes-only
        except (ImportError, AttributeError):
            continue

    raise ValueError(
        f"Indicator '{indicator_name}' not found. "
        f"Known indicators: {sorted(_INDICATOR_MAP.keys())}"
    )


def _call_indicator(func, arrays_spec: str, highs, lows, closes, volumes, opens=None):
    """Call an indicator function with the appropriate arrays based on spec.

    Returns the first array if the function returns a tuple.
    """
    args_map = {
        "c": (closes,),
        "v": (volumes,),
        "cv": (closes, volumes),
        "hl": (highs, lows),
        "hlc": (highs, lows, closes),
        "hlcv": (highs, lows, closes, volumes),
    }
    args = args_map.get(arrays_spec, (closes,))
    result = func(*args)

    # Some indicators return tuples (e.g., MACD returns line, signal, hist)
    if isinstance(result, tuple):
        return result[0]
    return result


# =========================================================================
# Core IC Computation
# =========================================================================

def compute_rolling_ic(
    indicator_values: np.ndarray,
    future_returns: np.ndarray,
    window: int = 252,
) -> np.ndarray:
    """Compute rolling Information Coefficient (Spearman rank correlation).

    IC[t] = spearman_corr(indicator[t-window:t], returns[t-window:t])

    Args:
        indicator_values: 1-D array of indicator values.
        future_returns: 1-D array of future returns (same length).
        window: Rolling window size.

    Returns:
        1-D array of IC values (NaN where insufficient data).
    """
    n = len(indicator_values)
    ic = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return ic

    for i in range(window - 1, n):
        start = i - window + 1
        ind_w = indicator_values[start:i + 1]
        ret_w = future_returns[start:i + 1]

        # Filter NaN from both arrays
        valid = ~(np.isnan(ind_w) | np.isnan(ret_w))
        if np.sum(valid) < max(10, window // 4):
            continue

        try:
            corr, _ = spearmanr(ind_w[valid], ret_w[valid])
            if not np.isnan(corr):
                ic[i] = corr
        except Exception:
            continue

    return ic


def detect_decay_alert(yearly_ics: dict, threshold: float = 0.05) -> list[str]:
    """Check if IC has been below threshold for 2+ consecutive years.

    Args:
        yearly_ics: dict mapping year (int or str) -> mean IC (float).
        threshold: IC threshold below which an indicator is considered weak.

    Returns:
        List of alert strings. Empty list = no decay detected.
    """
    alerts = []
    years = sorted(yearly_ics.keys())
    consecutive_low = 0
    low_start = None

    for year in years:
        ic_val = yearly_ics[year]
        if np.isnan(ic_val) or abs(ic_val) < threshold:
            if consecutive_low == 0:
                low_start = year
            consecutive_low += 1
        else:
            if consecutive_low >= 2:
                alerts.append(
                    f"IC below {threshold} for {consecutive_low} consecutive years "
                    f"({low_start}-{year - 1 if isinstance(year, int) else year})"
                )
            consecutive_low = 0
            low_start = None

    # Check trailing sequence
    if consecutive_low >= 2:
        alerts.append(
            f"IC below {threshold} for {consecutive_low} consecutive years "
            f"({low_start}-{years[-1]}), including the most recent data"
        )

    return alerts


def analyze_indicator_decay(
    indicator_name: str,
    symbols: list[str],
    window: int = 252,
    horizon: int = 5,
    start: str = "2015-01-01",
    end: str = "2026-03-01",
    freq: str = "daily",
) -> dict:
    """Full decay analysis for one indicator across multiple symbols.

    Args:
        indicator_name: Name of the indicator (e.g. "volume_momentum").
        symbols: List of trading symbols.
        window: Rolling IC window in bars.
        horizon: Forward return horizon in bars.
        start: Start date.
        end: End date.
        freq: Bar frequency.

    Returns:
        dict with keys: 'indicator', 'symbols', 'by_symbol', 'overall_yearly_ic',
        'alerts', 'summary'.
    """
    from alphaforge.data.market import MarketDataLoader
    from strategies.optimizer_core import map_freq, resample_bars
    from config import get_data_dir

    data_dir = get_data_dir()
    loader = MarketDataLoader(data_dir)
    load_freq, resample_factor = map_freq(freq)

    func, arrays_spec = _load_indicator_func(indicator_name)

    by_symbol = {}
    all_yearly = {}

    for sym in symbols:
        print(f"  [{sym}] Loading data...", end=" ", flush=True)
        try:
            bars = loader.load(sym, freq=load_freq, start=start, end=end)
            if bars is None or len(bars._close) < window + horizon + 50:
                print(f"insufficient data (need {window + horizon + 50}, got {len(bars._close) if bars else 0})")
                continue

            if resample_factor > 1:
                bars = resample_bars(bars, resample_factor)
                if bars is None or len(bars._close) < window + horizon + 50:
                    print("insufficient after resample")
                    continue

            closes = bars._close
            highs = bars._high
            lows = bars._low
            volumes = bars._volume
            bar_dates = bars._datetime if hasattr(bars, '_datetime') else None

            # Compute indicator
            ind_values = _call_indicator(func, arrays_spec, highs, lows, closes, volumes)
            if ind_values is None or len(ind_values) != len(closes):
                print("indicator length mismatch")
                continue

            # Compute forward returns
            future_returns = np.full(len(closes), np.nan, dtype=np.float64)
            for i in range(len(closes) - horizon):
                if closes[i] > 0:
                    future_returns[i] = (closes[i + horizon] - closes[i]) / closes[i]

            # Rolling IC
            ic_arr = compute_rolling_ic(ind_values, future_returns, window)

            # Yearly aggregation
            yearly_ic = {}
            if bar_dates is not None and len(bar_dates) > 0:
                import pandas as pd
                dates = pd.to_datetime(bar_dates)
                for year in range(dates.min().year, dates.max().year + 1):
                    mask = (dates.year == year) & (~np.isnan(ic_arr))
                    if np.sum(mask) > 20:
                        yearly_ic[year] = round(float(np.mean(ic_arr[mask])), 4)
            else:
                # Fallback: split by chunks
                valid_ic = ic_arr[~np.isnan(ic_arr)]
                if len(valid_ic) > 0:
                    yearly_ic[0] = round(float(np.mean(valid_ic)), 4)

            # Accumulate yearly ICs across symbols
            for yr, ic_val in yearly_ic.items():
                all_yearly.setdefault(yr, []).append(ic_val)

            # Symbol-level stats
            valid_ic = ic_arr[~np.isnan(ic_arr)]
            by_symbol[sym] = {
                "mean_ic": round(float(np.mean(valid_ic)), 4) if len(valid_ic) > 0 else np.nan,
                "std_ic": round(float(np.std(valid_ic)), 4) if len(valid_ic) > 0 else np.nan,
                "n_valid": int(len(valid_ic)),
                "yearly_ic": yearly_ic,
            }
            print(f"done (mean IC={by_symbol[sym]['mean_ic']:.4f})")

        except Exception as e:
            print(f"FAILED ({e})")
            continue

    # Aggregate overall yearly IC
    overall_yearly = {}
    for yr, vals in sorted(all_yearly.items()):
        overall_yearly[yr] = round(float(np.mean(vals)), 4)

    # Detect decay
    alerts = detect_decay_alert(overall_yearly)

    # Summary
    all_means = [s["mean_ic"] for s in by_symbol.values() if not np.isnan(s["mean_ic"])]
    overall_mean = round(float(np.mean(all_means)), 4) if all_means else np.nan

    if alerts:
        summary = f"ALERT: {indicator_name} shows alpha decay. {'; '.join(alerts)}"
    elif np.isnan(overall_mean):
        summary = f"{indicator_name}: insufficient data for decay analysis."
    elif overall_mean > 0.1:
        summary = f"{indicator_name}: strong predictive power (mean IC={overall_mean:.4f}). No decay detected."
    elif overall_mean > 0.05:
        summary = f"{indicator_name}: moderate predictive power (mean IC={overall_mean:.4f}). No decay detected."
    else:
        summary = f"{indicator_name}: weak predictive power (mean IC={overall_mean:.4f}). Monitor closely."

    return {
        "indicator": indicator_name,
        "symbols": symbols,
        "by_symbol": by_symbol,
        "overall_yearly_ic": overall_yearly,
        "alerts": alerts,
        "summary": summary,
    }


def _generate_decay_report(result: dict, output_path: str) -> str:
    """Generate a Markdown decay analysis report."""
    lines = [
        f"# Alpha Decay Analysis — {result['indicator']}",
        "",
        f"Symbols: {', '.join(result['symbols'])}",
        "",
        f"## Summary",
        result["summary"],
        "",
    ]

    # Alerts
    if result["alerts"]:
        lines.append("## Alerts")
        for alert in result["alerts"]:
            lines.append(f"- {alert}")
        lines.append("")

    # Per-symbol table
    lines += [
        "## Per-Symbol IC Statistics",
        "| Symbol | Mean IC | Std IC | Valid Bars |",
        "|--------|:------:|:-----:|:---------:|",
    ]
    for sym, stats in result["by_symbol"].items():
        mean_ic = f"{stats['mean_ic']:.4f}" if not np.isnan(stats["mean_ic"]) else "N/A"
        std_ic = f"{stats['std_ic']:.4f}" if not np.isnan(stats["std_ic"]) else "N/A"
        lines.append(f"| {sym} | {mean_ic} | {std_ic} | {stats['n_valid']} |")
    lines.append("")

    # Yearly IC table
    if result["overall_yearly_ic"]:
        lines += [
            "## Yearly IC (Cross-Symbol Average)",
            "| Year | Mean IC | Status |",
            "|:----:|:------:|:------:|",
        ]
        for yr, ic_val in sorted(result["overall_yearly_ic"].items()):
            if np.isnan(ic_val):
                status = "N/A"
            elif abs(ic_val) < 0.05:
                status = "WEAK"
            elif ic_val > 0.1:
                status = "Strong"
            else:
                status = "OK"
            lines.append(f"| {yr} | {ic_val:.4f} | {status} |")
        lines.append("")

    # Per-symbol yearly detail
    for sym, stats in result["by_symbol"].items():
        if stats["yearly_ic"]:
            lines.append(f"### {sym} — Yearly IC")
            lines.append("| Year | IC |")
            lines.append("|:----:|:--:|")
            for yr, ic_val in sorted(stats["yearly_ic"].items()):
                lines.append(f"| {yr} | {ic_val:.4f} |")
            lines.append("")

    lines += [
        "---",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Alpha Decay Detection — rolling IC analysis for indicators"
    )
    parser.add_argument("--indicator", required=True,
                        help="Indicator name (e.g. volume_momentum, rsi, adx)")
    parser.add_argument("--symbols", default="AG",
                        help="Comma-separated symbols (default: AG)")
    parser.add_argument("--window", type=int, default=252,
                        help="Rolling IC window in bars (default: 252)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in bars (default: 5)")
    parser.add_argument("--start", default="2015-01-01",
                        help="Start date (default: 2015-01-01)")
    parser.add_argument("--end", default="2026-03-01",
                        help="End date (default: 2026-03-01)")
    parser.add_argument("--freq", default="daily",
                        help="Bar frequency (default: daily)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: research_log/decay/)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    output_dir = args.output_dir or str(Path(QBASE_ROOT) / "research_log" / "decay")

    print(f"\n{'='*60}")
    print(f"Alpha Decay Analysis: {args.indicator}")
    print(f"Symbols: {', '.join(symbols)} | Window: {args.window} | Horizon: {args.horizon}")
    print(f"{'='*60}\n")

    result = analyze_indicator_decay(
        indicator_name=args.indicator,
        symbols=symbols,
        window=args.window,
        horizon=args.horizon,
        start=args.start,
        end=args.end,
        freq=args.freq,
    )

    report_path = str(Path(output_dir) / f"{args.indicator}.md")
    _generate_decay_report(result, report_path)

    print(f"\n{result['summary']}")
    print(f"Report: {report_path}\n")


if __name__ == "__main__":
    main()
