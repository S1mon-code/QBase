"""
QBase Screener — 多维度品种筛选
================================
用 QBase 指标对品种做特征排序，找到最适合某类策略的品种。

Usage:
    python screener/scanner.py --mode trend --start 2024-01-01 --end 2025-01-01 --top 10
    python screener/scanner.py --mode mean_reversion --start 2024-01-01 --top 15
"""
import sys
from pathlib import Path

# Ensure QBase and AlphaForge are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import get_alphaforge_path, get_data_dir

# Add AlphaForge to path
af_path = get_alphaforge_path()
if af_path not in sys.path:
    sys.path.insert(0, af_path)

from alphaforge.data.market import MarketDataLoader
from indicators.trend.adx import adx
from indicators.volatility.atr import atr
from indicators.volatility.historical_vol import historical_volatility
from indicators.momentum.roc import rate_of_change


def _load_data(loader, symbol, freq, start, end):
    """Load data for a symbol, return None if unavailable."""
    try:
        bars = loader.load(symbol, freq=freq, start=start, end=end)
        if bars is None or len(bars) < 60:
            return None
        return bars
    except Exception:
        return None


def _extract_arrays(bars):
    """Extract OHLCV arrays from BarArray."""
    return (
        np.array(bars.high, dtype=float),
        np.array(bars.low, dtype=float),
        np.array(bars.close, dtype=float),
        np.array(bars.volume, dtype=float),
    )


def scan_trend(loader, symbols, freq="daily", start=None, end=None):
    """Find symbols with strongest trends (high ADX + directional momentum)."""
    results = []
    for sym in symbols:
        bars = _load_data(loader, sym, freq, start, end)
        if bars is None:
            continue
        h, l, c, v = _extract_arrays(bars)
        adx_vals = adx(h, l, c, period=14)
        roc_vals = rate_of_change(c, 20)
        mean_adx = np.nanmean(adx_vals[-60:])
        abs_roc = np.nanmean(np.abs(roc_vals[-60:]))
        results.append({
            "symbol": sym,
            "mean_adx_60d": round(mean_adx, 2),
            "mean_abs_roc_60d": round(abs_roc, 2),
            "trend_score": round(mean_adx * abs_roc, 2),
        })
    df = pd.DataFrame(results)
    return df.sort_values("trend_score", ascending=False).reset_index(drop=True)


def scan_mean_reversion(loader, symbols, freq="daily", start=None, end=None):
    """Find symbols suited for mean-reversion (high vol + low ADX = range-bound)."""
    results = []
    for sym in symbols:
        bars = _load_data(loader, sym, freq, start, end)
        if bars is None:
            continue
        h, l, c, v = _extract_arrays(bars)
        adx_vals = adx(h, l, c, period=14)
        hvol = historical_volatility(c, period=20)
        mean_adx = np.nanmean(adx_vals[-60:])
        mean_vol = np.nanmean(hvol[-60:])
        # High vol + low ADX = good for mean reversion
        mr_score = mean_vol / max(mean_adx, 1.0)
        results.append({
            "symbol": sym,
            "mean_adx_60d": round(mean_adx, 2),
            "mean_hvol_60d": round(mean_vol, 4),
            "mr_score": round(mr_score, 4),
        })
    df = pd.DataFrame(results)
    return df.sort_values("mr_score", ascending=False).reset_index(drop=True)


def scan_breakout(loader, symbols, freq="daily", start=None, end=None):
    """Find symbols with volatility contraction (potential breakout candidates)."""
    results = []
    for sym in symbols:
        bars = _load_data(loader, sym, freq, start, end)
        if bars is None:
            continue
        h, l, c, v = _extract_arrays(bars)
        atr_vals = atr(h, l, c, period=14)
        hvol = historical_volatility(c, period=20)
        # Compare recent vol to longer-term vol
        recent_vol = np.nanmean(hvol[-20:])
        longer_vol = np.nanmean(hvol[-60:])
        vol_ratio = recent_vol / max(longer_vol, 1e-8)
        recent_atr = np.nanmean(atr_vals[-20:])
        results.append({
            "symbol": sym,
            "recent_vol": round(recent_vol, 4),
            "longer_vol": round(longer_vol, 4),
            "vol_contraction": round(vol_ratio, 4),
            "recent_atr": round(recent_atr, 2),
        })
    df = pd.DataFrame(results)
    # Lower ratio = more contraction = better breakout candidate
    return df.sort_values("vol_contraction", ascending=True).reset_index(drop=True)


def scan_volatility(loader, symbols, freq="daily", start=None, end=None):
    """Rank symbols by volatility level."""
    results = []
    for sym in symbols:
        bars = _load_data(loader, sym, freq, start, end)
        if bars is None:
            continue
        h, l, c, v = _extract_arrays(bars)
        hvol = historical_volatility(c, period=20)
        atr_vals = atr(h, l, c, period=14)
        mean_vol = np.nanmean(hvol[-60:])
        mean_atr = np.nanmean(atr_vals[-60:])
        # Normalize ATR by price
        atr_pct = mean_atr / max(np.nanmean(c[-60:]), 1e-8) * 100
        results.append({
            "symbol": sym,
            "mean_hvol_60d": round(mean_vol, 4),
            "atr_pct_60d": round(atr_pct, 4),
        })
    df = pd.DataFrame(results)
    return df.sort_values("mean_hvol_60d", ascending=False).reset_index(drop=True)


def rank_by_indicator(indicator_func, loader, symbols, freq="daily",
                      start=None, end=None, **kwargs):
    """Generic: rank symbols by any indicator's latest value."""
    results = []
    for sym in symbols:
        bars = _load_data(loader, sym, freq, start, end)
        if bars is None:
            continue
        h, l, c, v = _extract_arrays(bars)
        try:
            val = indicator_func(c, **kwargs)
            latest = val[-1] if not np.isnan(val[-1]) else np.nanmean(val[-20:])
            results.append({"symbol": sym, "value": round(latest, 4)})
        except Exception:
            continue
    df = pd.DataFrame(results)
    return df.sort_values("value", ascending=False).reset_index(drop=True)


SCAN_MODES = {
    "trend": scan_trend,
    "mean_reversion": scan_mean_reversion,
    "breakout": scan_breakout,
    "volatility": scan_volatility,
}


def scan(mode="trend", symbols=None, freq="daily", start=None, end=None, top_n=20):
    """Main entry: scan symbols by mode, return top N."""
    loader = MarketDataLoader(get_data_dir())
    if symbols is None:
        symbols = loader.available_symbols()
    scan_func = SCAN_MODES.get(mode)
    if scan_func is None:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(SCAN_MODES.keys())}")
    df = scan_func(loader, symbols, freq, start, end)
    return df.head(top_n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QBase Screener")
    parser.add_argument("--mode", default="trend",
                        choices=list(SCAN_MODES.keys()),
                        help="Scan mode")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--freq", default="daily", help="Data frequency")
    parser.add_argument("--top", type=int, default=20, help="Show top N results")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    syms = args.symbols.split(",") if args.symbols else None
    result = scan(mode=args.mode, symbols=syms, freq=args.freq,
                  start=args.start, end=args.end, top_n=args.top)
    print(f"\n=== QBase Screener: {args.mode.upper()} (top {args.top}) ===\n")
    print(result.to_string(index=False))
    print()
