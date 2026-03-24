import numpy as np


def choppiness_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Choppiness Index (Bill Dreiss).

    CI = 100 * LOG10(SUM(ATR(1), period) / (highest_high - lowest_low)) / LOG10(period)

    Measures whether the market is trending or choppy.
    Range [0, 100]. High (>61.8) = choppy/sideways, Low (<38.2) = trending.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= period:
        return np.full(n, np.nan)

    # True Range (ATR with period=1 is just TR)
    tr = np.full(n, np.nan)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    result = np.full(n, np.nan)
    log_period = np.log10(period)

    for i in range(period, n):
        atr_sum = tr[i - period + 1:i + 1].sum()
        highest = highs[i - period + 1:i + 1].max()
        lowest = lows[i - period + 1:i + 1].min()
        hl_range = highest - lowest

        if hl_range == 0 or atr_sum == 0:
            result[i] = np.nan
        else:
            result[i] = 100.0 * np.log10(atr_sum / hl_range) / log_period

    return result
