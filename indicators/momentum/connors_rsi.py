import numpy as np


def _rsi(data: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder's smoothing."""
    if data.size <= period:
        return np.full(data.size, np.nan)

    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(data.size, np.nan)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    alpha = 1.0 / period
    for i in range(period, len(deltas)):
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return result


def connors_rsi(
    closes: np.ndarray,
    rsi_period: int = 3,
    streak_period: int = 2,
    pct_rank_period: int = 100,
) -> np.ndarray:
    """Connors RSI.

    CRSI = (RSI(close, rsi_period) + RSI(streak, streak_period) + PercentRank) / 3

    Components:
    1. Short-term RSI of price
    2. RSI of consecutive up/down streak length
    3. Percent rank of current price change over lookback window

    Range [0, 100].
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= max(rsi_period, streak_period) + 1:
        return np.full(n, np.nan)

    # Component 1: RSI of price
    rsi_price = _rsi(closes, rsi_period)

    # Component 2: Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif closes[i] < closes[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0

    rsi_streak = _rsi(streak, streak_period)

    # Component 3: Percent rank of today's price change
    pct_changes = np.diff(closes)
    pct_rank = np.full(n, np.nan)
    for i in range(pct_rank_period, len(pct_changes)):
        current = pct_changes[i]
        lookback = pct_changes[i - pct_rank_period:i]
        pct_rank[i + 1] = 100.0 * np.sum(lookback < current) / pct_rank_period

    # Composite — average available components, NaN where none are valid
    components = np.array([rsi_price, rsi_streak, pct_rank])
    with np.errstate(all='ignore'):
        result = np.nanmean(components, axis=0)
    # Where ALL three are NaN, keep NaN
    all_nan = np.all(np.isnan(components), axis=0)
    result[all_nan] = np.nan
    # Where all three are available, use strict average
    all_valid = np.all(~np.isnan(components), axis=0)
    result[all_valid] = (rsi_price[all_valid] + rsi_streak[all_valid] + pct_rank[all_valid]) / 3.0

    return result
