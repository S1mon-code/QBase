import numpy as np


def chandelier_exit(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 22,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Chandelier Exit (Chuck LeBeau).

    ATR-based trailing stops hung from the highest high (for longs) and
    lowest low (for shorts) over a lookback period.

    Formula:
        Long Exit  = Highest High(period) - multiplier * ATR(period)
        Short Exit = Lowest Low(period) + multiplier * ATR(period)

    Returns:
        (long_exit, short_exit) — both np.ndarray

    Reference: Charles Le Beau; featured in Alexander Elder's
    "Come Into My Trading Room" (2002). Default: 22-period, 3x ATR.
    """
    n = len(closes)
    long_exit = np.full(n, np.nan)
    short_exit = np.full(n, np.nan)

    if n == 0 or n < period + 1:
        return long_exit, short_exit

    # True Range
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR using Wilder's smoothing
    atr_vals = np.full(n, np.nan)
    atr_vals[period] = np.mean(tr[1 : period + 1])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr_vals[i] = atr_vals[i - 1] * (1.0 - alpha) + tr[i] * alpha

    # Chandelier levels
    for i in range(period, n):
        if np.isnan(atr_vals[i]):
            continue
        highest = np.max(highs[i - period + 1 : i + 1])
        lowest = np.min(lows[i - period + 1 : i + 1])
        long_exit[i] = highest - multiplier * atr_vals[i]
        short_exit[i] = lowest + multiplier * atr_vals[i]

    return long_exit, short_exit
