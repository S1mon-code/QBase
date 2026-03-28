"""Regime attribution — tag trades with market regime and compute per-regime stats."""
import sys
from pathlib import Path
from dataclasses import dataclass, field

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@dataclass
class RegimeStats:
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0
    avg_holding_bars: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


@dataclass
class RegimeAttributionResult:
    strategy_version: str
    symbol: str
    period: str
    total_trades: int
    total_sharpe: float
    by_trend: dict = field(default_factory=dict)
    by_volatility: dict = field(default_factory=dict)
    by_activity: dict = field(default_factory=dict)
    cross_trend_vol: dict = field(default_factory=dict)
    best_regime: str = ""
    worst_regime: str = ""


def pair_trades(trades_df: pd.DataFrame) -> list[dict]:
    """Pair entry and exit trades into round-trip records.

    Walks through trades chronologically, matching opens (buy) with closes (sell)
    for long trades. Handles partial closes by splitting into sub-trades.
    """
    if trades_df is None or len(trades_df) == 0:
        return []

    pairs = []
    open_lots = 0
    open_price = 0.0
    open_dt = None
    open_side = 0  # 1=long, -1=short

    for _, row in trades_df.iterrows():
        raw_side = row.get('side', '')
        # Handle both string ('buy'/'sell') and numeric (1/-1) side formats
        if isinstance(raw_side, (int, float)):
            side_str = 'buy' if int(raw_side) == 1 else 'sell'
        else:
            side_str = str(raw_side)
        lots = int(row.get('lots', 0))
        price = float(row.get('price', 0))
        dt = row.get('datetime', '')

        if side_str == 'buy':
            if open_lots == 0:
                open_lots = lots
                open_price = price
                open_dt = dt
                open_side = 1
            elif open_side == 1:
                total_cost = open_price * open_lots + price * lots
                open_lots += lots
                open_price = total_cost / open_lots if open_lots > 0 else price
            elif open_side == -1:
                close_lots = min(lots, open_lots)
                pnl_pct = (open_price - price) / open_price if open_price > 0 else 0
                pairs.append({
                    'entry_datetime': open_dt,
                    'exit_datetime': dt,
                    'side': -1,
                    'entry_price': open_price,
                    'exit_price': price,
                    'lots': close_lots,
                    'pnl_pct': pnl_pct,
                    'holding_bars': 0,
                })
                open_lots -= close_lots
                if open_lots <= 0:
                    open_lots = 0
                    open_side = 0

        elif side_str == 'sell':
            if open_lots > 0 and open_side == 1:
                close_lots = min(lots, open_lots)
                pnl_pct = (price - open_price) / open_price if open_price > 0 else 0
                pairs.append({
                    'entry_datetime': open_dt,
                    'exit_datetime': dt,
                    'side': 1,
                    'entry_price': open_price,
                    'exit_price': price,
                    'lots': close_lots,
                    'pnl_pct': pnl_pct,
                    'holding_bars': 0,
                })
                open_lots -= close_lots
                if open_lots <= 0:
                    open_lots = 0
                    open_side = 0
            elif open_lots == 0:
                open_lots = lots
                open_price = price
                open_dt = dt
                open_side = -1

    return pairs


def _rolling_percentile(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling percentile rank of each value within its trailing window."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        w = arr[i - window:i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) < 2:
            continue
        rank = np.sum(valid < arr[i]) / (len(valid) - 1)
        out[i] = rank
    return out


def _compute_regime_labels(highs, lows, closes, volumes):
    """Compute regime labels for every bar across three dimensions."""
    from indicators.trend.adx import adx
    from indicators.volatility.atr import atr
    from indicators.trend.sma import sma

    n = len(closes)
    adx_arr = adx(highs, lows, closes, period=14)
    atr_arr = atr(highs, lows, closes, period=14)
    vol_ma = sma(volumes.astype(np.float64), 20)

    trend_labels = np.full(n, 'unknown', dtype=object)
    for i in range(n):
        if np.isnan(adx_arr[i]):
            trend_labels[i] = 'unknown'
        elif adx_arr[i] > 25:
            trend_labels[i] = 'strong'
        elif adx_arr[i] > 15:
            trend_labels[i] = 'weak'
        else:
            trend_labels[i] = 'none'

    atr_pctile = _rolling_percentile(atr_arr, 252)
    vol_labels = np.full(n, 'unknown', dtype=object)
    for i in range(n):
        if np.isnan(atr_pctile[i]):
            vol_labels[i] = 'unknown'
        elif atr_pctile[i] > 0.75:
            vol_labels[i] = 'high'
        elif atr_pctile[i] > 0.25:
            vol_labels[i] = 'normal'
        else:
            vol_labels[i] = 'low'

    activity_labels = np.full(n, 'unknown', dtype=object)
    for i in range(n):
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            activity_labels[i] = 'unknown'
        else:
            ratio = volumes[i] / vol_ma[i]
            if ratio > 1.5:
                activity_labels[i] = 'active'
            elif ratio > 0.7:
                activity_labels[i] = 'normal'
            else:
                activity_labels[i] = 'quiet'

    return trend_labels, vol_labels, activity_labels


def _compute_regime_stats(trades_in_regime: list[dict]) -> RegimeStats:
    """Compute stats for a group of trades."""
    if not trades_in_regime:
        return RegimeStats()

    pnls = [t['pnl_pct'] for t in trades_in_regime]
    wins = [p for p in pnls if p > 0]
    holdings = [t.get('holding_bars', 0) for t in trades_in_regime]

    return RegimeStats(
        n_trades=len(pnls),
        win_rate=round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        avg_pnl_pct=round(float(np.mean(pnls)) * 100, 2) if pnls else 0,
        total_pnl_pct=round(float(np.sum(pnls)) * 100, 2) if pnls else 0,
        avg_holding_bars=round(float(np.mean(holdings)), 1) if holdings else 0,
        best_trade_pnl=round(float(max(pnls)) * 100, 2) if pnls else 0,
        worst_trade_pnl=round(float(min(pnls)) * 100, 2) if pnls else 0,
    )


def _match_datetime_to_bar_index(dt, bar_datetimes):
    """Find the bar index closest to (and not after) the given datetime."""
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    indices = np.where(bar_datetimes <= dt)[0]
    if len(indices) == 0:
        return 0
    return indices[-1]


def run_regime_attribution(
    strategy_cls,
    params: dict,
    symbol: str,
    start: str,
    end: str,
    freq: str = "daily",
    data_dir: str | None = None,
) -> RegimeAttributionResult:
    """Run strategy, tag each trade with market regime, compute per-regime stats."""
    from strategies.optimizer_core import create_strategy_with_params
    from attribution.signal import run_backtest_full
    from alphaforge.data.market import MarketDataLoader
    from strategies.optimizer_core import map_freq, resample_bars

    version = getattr(strategy_cls, 'name', str(strategy_cls))

    strategy = create_strategy_with_params(strategy_cls, params)
    result = run_backtest_full(strategy, symbol, start, end, freq, data_dir)
    if result is None or result.trades is None or len(result.trades) == 0:
        return RegimeAttributionResult(
            strategy_version=version, symbol=symbol,
            period=f"{start} ~ {end}", total_trades=0, total_sharpe=-999,
        )

    if data_dir is None:
        from config import get_data_dir
        data_dir = get_data_dir()
    loader = MarketDataLoader(data_dir)
    load_freq, resample_factor = map_freq(freq)
    bars = loader.load(symbol, freq=load_freq, start=start, end=end)
    if resample_factor > 1:
        bars = resample_bars(bars, resample_factor)

    closes = bars._close
    highs = bars._high
    lows = bars._low
    volumes = bars._volume
    bar_datetimes = bars._datetime if hasattr(bars, '_datetime') else np.arange(len(closes))

    trend_labels, vol_labels, activity_labels = _compute_regime_labels(
        highs, lows, closes, volumes,
    )

    paired = pair_trades(result.trades)
    if not paired:
        return RegimeAttributionResult(
            strategy_version=version, symbol=symbol,
            period=f"{start} ~ {end}", total_trades=0,
            total_sharpe=float(result.sharpe),
        )

    for trade in paired:
        idx = _match_datetime_to_bar_index(trade['entry_datetime'], bar_datetimes)
        trade['trend_regime'] = str(trend_labels[idx])
        trade['vol_regime'] = str(vol_labels[idx])
        trade['activity_regime'] = str(activity_labels[idx])

    by_trend = {}
    by_vol = {}
    by_act = {}
    cross = {}

    for label in ['strong', 'weak', 'none', 'unknown']:
        group = [t for t in paired if t.get('trend_regime') == label]
        if group:
            by_trend[label] = _compute_regime_stats(group)

    for label in ['high', 'normal', 'low', 'unknown']:
        group = [t for t in paired if t.get('vol_regime') == label]
        if group:
            by_vol[label] = _compute_regime_stats(group)

    for label in ['active', 'normal', 'quiet', 'unknown']:
        group = [t for t in paired if t.get('activity_regime') == label]
        if group:
            by_act[label] = _compute_regime_stats(group)

    for tl in ['strong', 'weak', 'none']:
        for vl in ['high', 'normal', 'low']:
            group = [t for t in paired
                     if t.get('trend_regime') == tl and t.get('vol_regime') == vl]
            if group:
                cross[(tl, vl)] = _compute_regime_stats(group)

    all_regimes = {}
    for tl, stats in by_trend.items():
        if stats.n_trades >= 2:
            all_regimes[f"trend={tl}"] = stats.avg_pnl_pct
    for vl, stats in by_vol.items():
        if stats.n_trades >= 2:
            all_regimes[f"vol={vl}"] = stats.avg_pnl_pct
    for al, stats in by_act.items():
        if stats.n_trades >= 2:
            all_regimes[f"activity={al}"] = stats.avg_pnl_pct

    best = max(all_regimes, key=all_regimes.get) if all_regimes else ""
    worst = min(all_regimes, key=all_regimes.get) if all_regimes else ""

    return RegimeAttributionResult(
        strategy_version=version,
        symbol=symbol,
        period=f"{start} ~ {end}",
        total_trades=len(paired),
        total_sharpe=round(float(result.sharpe), 3),
        by_trend=by_trend,
        by_volatility=by_vol,
        by_activity=by_act,
        cross_trend_vol=cross,
        best_regime=best,
        worst_regime=worst,
    )
