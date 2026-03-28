"""Drawdown Attribution — identify worst drawdown and attribute to strategies + regimes.

Usage:
    python -m attribution.drawdown \
        --weights strategies/strong_trend/portfolio/weights_ag.json \
        --start 2025-01-01 --end 2026-03-01
"""
import sys
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class DrawdownPeriod:
    start_date: str
    end_date: str
    duration_days: int
    drawdown_pct: float  # negative
    peak_equity: float
    trough_equity: float


@dataclass
class DrawdownAttribution:
    period: DrawdownPeriod
    strategy_contributions: dict  # {version: {"pnl": float, "pct_of_dd": float}}
    regime_during_dd: dict  # {"trend": "weak", "volume": "quiet", "volatility": "low"}
    conclusion: str  # auto-generated one-line summary


def find_max_drawdown(equity_curve: np.ndarray, dates=None) -> DrawdownPeriod:
    """Find the maximum drawdown period from an equity curve.

    Args:
        equity_curve: 1-D array of equity values.
        dates: Optional array of date strings or timestamps aligned with equity_curve.

    Returns:
        DrawdownPeriod with peak-to-trough information.
    """
    if len(equity_curve) < 2:
        d = dates[0] if dates is not None and len(dates) > 0 else "N/A"
        return DrawdownPeriod(
            start_date=str(d), end_date=str(d),
            duration_days=0, drawdown_pct=0.0,
            peak_equity=float(equity_curve[0]) if len(equity_curve) > 0 else 0.0,
            trough_equity=float(equity_curve[0]) if len(equity_curve) > 0 else 0.0,
        )

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / np.where(running_max > 0, running_max, 1.0)

    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(equity_curve[:trough_idx + 1]))

    dd_pct = float(drawdowns[trough_idx])
    peak_val = float(equity_curve[peak_idx])
    trough_val = float(equity_curve[trough_idx])

    if dates is not None and len(dates) > 0:
        start_str = str(dates[peak_idx])[:10]
        end_str = str(dates[trough_idx])[:10]
        try:
            d0 = pd.Timestamp(dates[peak_idx])
            d1 = pd.Timestamp(dates[trough_idx])
            duration = (d1 - d0).days
        except Exception:
            duration = trough_idx - peak_idx
    else:
        start_str = str(peak_idx)
        end_str = str(trough_idx)
        duration = trough_idx - peak_idx

    return DrawdownPeriod(
        start_date=start_str,
        end_date=end_str,
        duration_days=max(0, duration),
        drawdown_pct=round(dd_pct * 100, 2),
        peak_equity=round(peak_val, 2),
        trough_equity=round(trough_val, 2),
    )


def _compute_regime_during_period(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict:
    """Compute average market regime indicators during a specific period.

    Computes ADX, volume ratio, and ATR percentile directly on the price
    arrays (lighter than the full regime attribution pipeline).
    """
    from indicators.trend.adx import adx
    from indicators.volatility.atr import atr
    from indicators.trend.sma import sma

    adx_arr = adx(highs, lows, closes, period=14)
    atr_arr = atr(highs, lows, closes, period=14)
    vol_ma = sma(volumes.astype(np.float64), 20)

    # Slice to the drawdown period
    sl = slice(start_idx, end_idx + 1)

    # ADX average
    adx_slice = adx_arr[sl]
    adx_valid = adx_slice[~np.isnan(adx_slice)]
    adx_avg = float(np.mean(adx_valid)) if len(adx_valid) > 0 else np.nan

    if np.isnan(adx_avg):
        trend_label = "unknown"
    elif adx_avg > 25:
        trend_label = "strong"
    elif adx_avg > 15:
        trend_label = "weak"
    else:
        trend_label = "none"

    # Volume ratio (current / 20-day MA)
    vol_slice = volumes[sl].astype(np.float64)
    vol_ma_slice = vol_ma[sl]
    valid_mask = (~np.isnan(vol_ma_slice)) & (vol_ma_slice > 0)
    if np.any(valid_mask):
        vol_ratio = float(np.mean(vol_slice[valid_mask] / vol_ma_slice[valid_mask]))
    else:
        vol_ratio = np.nan

    if np.isnan(vol_ratio):
        volume_label = "unknown"
    elif vol_ratio > 1.5:
        volume_label = "active"
    elif vol_ratio > 0.7:
        volume_label = "normal"
    else:
        volume_label = "quiet"

    # ATR percentile (rolling 252-bar)
    atr_full = atr_arr[:end_idx + 1]
    atr_valid = atr_full[~np.isnan(atr_full)]
    if len(atr_valid) > 10:
        atr_dd = atr_arr[sl]
        atr_dd_valid = atr_dd[~np.isnan(atr_dd)]
        if len(atr_dd_valid) > 0:
            atr_dd_mean = float(np.mean(atr_dd_valid))
            atr_pctile = float(np.sum(atr_valid < atr_dd_mean) / len(atr_valid) * 100)
        else:
            atr_pctile = np.nan
    else:
        atr_pctile = np.nan

    if np.isnan(atr_pctile):
        vol_label = "unknown"
    elif atr_pctile > 75:
        vol_label = "high"
    elif atr_pctile > 25:
        vol_label = "normal"
    else:
        vol_label = "low"

    return {
        "trend": trend_label,
        "trend_detail": f"ADX avg = {adx_avg:.1f}" if not np.isnan(adx_avg) else "N/A",
        "volume": volume_label,
        "volume_detail": f"avg {vol_ratio:.1f}x of 20-day mean" if not np.isnan(vol_ratio) else "N/A",
        "volatility": vol_label,
        "volatility_detail": f"ATR percentile {atr_pctile:.0f}%" if not np.isnan(atr_pctile) else "N/A",
    }


def _generate_report(
    attribution: DrawdownAttribution,
    symbol: str,
    output_path: str,
) -> str:
    """Generate a Markdown drawdown attribution report."""
    p = attribution.period
    lines = [
        f"# Portfolio Drawdown Attribution — {symbol}",
        "",
        "## Maximum Drawdown",
        f"- Period: {p.start_date} ~ {p.end_date} ({p.duration_days} days)",
        f"- Magnitude: {p.drawdown_pct:+.2f}%",
        f"- Peak equity: ¥{p.peak_equity:,.0f} → Trough: ¥{p.trough_equity:,.0f}",
        "",
        "## Strategy Contributions During Drawdown",
        "| Strategy | Weight | PnL During DD | % of Total DD |",
        "|----------|:------:|:------------:|:-------------:|",
    ]

    for ver, info in sorted(
        attribution.strategy_contributions.items(),
        key=lambda x: x[1].get("pnl", 0),
    ):
        weight_str = f"{info.get('weight', 0) * 100:.0f}%"
        pnl = info.get("pnl", 0)
        pct = info.get("pct_of_dd", 0)
        lines.append(f"| {ver} | {weight_str} | ¥{pnl:,.0f} | {pct:.0f}% |")

    regime = attribution.regime_during_dd
    lines += [
        "",
        "## Market Regime During Drawdown",
        f"- Trend: {regime.get('trend', 'N/A').capitalize()} ({regime.get('trend_detail', '')})",
        f"- Volume: {regime.get('volume', 'N/A').capitalize()} ({regime.get('volume_detail', '')})",
        f"- Volatility: {regime.get('volatility', 'N/A').capitalize()} ({regime.get('volatility_detail', '')})",
        "",
        "## Conclusion",
        attribution.conclusion,
        "",
        "---",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def attribute_drawdown(
    weights_path: str,
    symbol: str,
    start: str,
    end: str,
    output_dir: str = None,
) -> DrawdownAttribution:
    """Full drawdown attribution pipeline.

    1. Load weights JSON -> get strategy list
    2. Run backtest for each strategy individually
    3. Compute portfolio equity curve (weighted sum of individual equity curves)
    4. Find the maximum drawdown period
    5. For each strategy: compute its PnL during the DD period -> attribution %
    6. Tag the DD period with market regime
    7. Output markdown report
    """
    from attribution.batch import parse_weights_json, load_strategy_class, _detect_strategy_dir
    from attribution.signal import run_backtest_full
    from strategies.optimizer_core import create_strategy_with_params, map_freq, resample_bars
    from alphaforge.data.market import MarketDataLoader
    from config import get_data_dir

    if output_dir is None:
        output_dir = str(Path(QBASE_ROOT) / "research_log" / "attribution")

    entries, meta = parse_weights_json(weights_path)
    strategy_dir = _detect_strategy_dir(weights_path)

    if not symbol and meta.get("symbol"):
        symbol = meta["symbol"]

    print(f"\n{'='*60}")
    print(f"Drawdown Attribution: {len(entries)} strategies")
    print(f"Symbol: {symbol} | Period: {start} ~ {end}")
    print(f"{'='*60}\n")

    # --- Step 1-2: Run backtests, collect equity curves ---
    equity_curves = {}  # version -> np.array of equity values
    equity_dates = None  # shared date index (from first successful run)

    for entry in entries:
        ver = entry.version
        if not entry.params:
            print(f"  [{ver}] SKIP — no params")
            continue

        try:
            print(f"  [{ver}] Running backtest...", end=" ", flush=True)
            strategy_cls = load_strategy_class(ver, strategy_dir)
            strategy = create_strategy_with_params(strategy_cls, entry.params)
            result = run_backtest_full(strategy, symbol, start, end, freq=entry.freq)

            if result is None:
                print("no result")
                continue

            eq = getattr(result, "equity_curve", None)
            if eq is None:
                eq = getattr(result, "equity", None)
            if eq is None or len(eq) == 0:
                print("no equity curve")
                continue

            equity_curves[ver] = np.array(eq, dtype=np.float64)

            # Try to get dates from result
            if equity_dates is None:
                dt = getattr(result, "dates", None)
                if dt is None:
                    dt = getattr(result, "datetime", None)
                if dt is not None and len(dt) > 0:
                    equity_dates = dt

            print(f"done (len={len(eq)})")
        except Exception as e:
            print(f"FAILED ({e})")

    if not equity_curves:
        raise ValueError("No strategies produced equity curves — cannot compute drawdown.")

    # --- Step 3: Portfolio equity curve (weighted sum of individual returns) ---
    # Align all curves to the same length (shortest)
    min_len = min(len(ec) for ec in equity_curves.values())
    weights_map = {e.version: e.weight for e in entries}
    initial_capital = 1_000_000

    # Compute portfolio equity from weighted returns
    portfolio_equity = np.full(min_len, float(initial_capital))
    for ver, ec in equity_curves.items():
        ec_trimmed = ec[:min_len]
        # Strategy returns relative to starting equity
        returns = np.diff(ec_trimmed) / np.where(ec_trimmed[:-1] != 0, ec_trimmed[:-1], 1.0)
        w = weights_map.get(ver, 0)
        for i in range(1, min_len):
            portfolio_equity[i] = portfolio_equity[i - 1] * (1 + w * returns[i - 1]) \
                if i - 1 < len(returns) else portfolio_equity[i - 1]

    if equity_dates is not None:
        equity_dates = equity_dates[:min_len]

    # --- Step 4: Find max drawdown ---
    dd_period = find_max_drawdown(portfolio_equity, dates=equity_dates)

    # Map dates back to indices for the equity arrays
    if equity_dates is not None:
        try:
            date_series = pd.Series(range(len(equity_dates)),
                                    index=pd.to_datetime(equity_dates))
            peak_dt = pd.Timestamp(dd_period.start_date)
            trough_dt = pd.Timestamp(dd_period.end_date)
            # Find closest indices
            peak_idx = int(date_series.index.searchsorted(peak_dt))
            trough_idx = int(date_series.index.searchsorted(trough_dt))
            peak_idx = min(peak_idx, min_len - 1)
            trough_idx = min(trough_idx, min_len - 1)
        except Exception:
            # Fallback: re-derive from equity curve
            running_max = np.maximum.accumulate(portfolio_equity)
            drawdowns = (portfolio_equity - running_max) / np.where(running_max > 0, running_max, 1.0)
            trough_idx = int(np.argmin(drawdowns))
            peak_idx = int(np.argmax(portfolio_equity[:trough_idx + 1]))
    else:
        running_max = np.maximum.accumulate(portfolio_equity)
        drawdowns = (portfolio_equity - running_max) / np.where(running_max > 0, running_max, 1.0)
        trough_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(portfolio_equity[:trough_idx + 1]))

    # --- Step 5: Strategy contributions during DD period ---
    total_dd_pnl = float(portfolio_equity[trough_idx] - portfolio_equity[peak_idx])
    strategy_contributions = {}

    for ver, ec in equity_curves.items():
        ec_trimmed = ec[:min_len]
        w = weights_map.get(ver, 0)
        # Strategy's PnL during the DD period, scaled by weight
        strat_pnl_during_dd = float(ec_trimmed[trough_idx] - ec_trimmed[peak_idx]) * w
        pct_of_dd = (strat_pnl_during_dd / total_dd_pnl * 100) if total_dd_pnl != 0 else 0
        strategy_contributions[ver] = {
            "pnl": round(strat_pnl_during_dd, 2),
            "pct_of_dd": round(pct_of_dd, 1),
            "weight": w,
        }

    # --- Step 6: Regime tagging during DD period ---
    data_dir = get_data_dir()
    loader = MarketDataLoader(data_dir)
    load_freq, resample_factor = map_freq("daily")
    bars = loader.load(symbol, freq=load_freq, start=start, end=end)
    if resample_factor > 1:
        bars = resample_bars(bars, resample_factor)

    # Map DD equity indices to price bar indices (approximate by ratio)
    n_bars = len(bars._close)
    bar_peak_idx = int(peak_idx / min_len * n_bars)
    bar_trough_idx = int(trough_idx / min_len * n_bars)
    bar_peak_idx = min(bar_peak_idx, n_bars - 1)
    bar_trough_idx = min(bar_trough_idx, n_bars - 1)

    regime = _compute_regime_during_period(
        bars._high, bars._low, bars._close, bars._volume,
        bar_peak_idx, bar_trough_idx,
    )

    # --- Step 7: Conclusion ---
    freq_counts = {}
    for entry in entries:
        if entry.version in equity_curves:
            freq_counts.setdefault(entry.freq, []).append(entry.version)

    # Find strategy with smallest loss
    if strategy_contributions:
        best_ver = min(strategy_contributions, key=lambda v: abs(strategy_contributions[v]["pnl"]))
        best_freq = next((e.freq for e in entries if e.version == best_ver), "daily")
    else:
        best_ver = "N/A"
        best_freq = "N/A"

    conclusion = (
        f"Drawdown occurred during {regime['trend']}-trend + {regime['volume']}-volume regime. "
        f"{best_ver} ({best_freq} freq) had smallest loss — "
        f"frequency diversification {'helped' if len(freq_counts) > 1 else 'was not applicable'}."
    )

    result = DrawdownAttribution(
        period=dd_period,
        strategy_contributions=strategy_contributions,
        regime_during_dd=regime,
        conclusion=conclusion,
    )

    # Write report
    report_path = str(Path(output_dir) / f"drawdown_{symbol}.md")
    _generate_report(result, symbol, report_path)
    print(f"\nReport: {report_path}")

    return result


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drawdown Attribution — identify worst drawdown and attribute to strategies"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to portfolio weights JSON")
    parser.add_argument("--symbol", default="",
                        help="Override symbol (default: auto-detect from weights JSON)")
    parser.add_argument("--start", default="2025-01-01",
                        help="Backtest start date (default: 2025-01-01)")
    parser.add_argument("--end", default="2026-03-01",
                        help="Backtest end date (default: 2026-03-01)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: research_log/attribution/)")
    args = parser.parse_args()

    symbol = args.symbol
    if not symbol:
        import json
        with open(args.weights) as f:
            data = json.load(f)
        symbol = data.get("meta", {}).get("symbol", "AG")

    attribute_drawdown(
        weights_path=args.weights,
        symbol=symbol,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
