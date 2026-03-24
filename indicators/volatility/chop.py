import numpy as np


def atr_ratio(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    short_period: int = 5,
    long_period: int = 20,
) -> np.ndarray:
    """ATR Ratio — Choppiness proxy using short/long ATR.

    Compares short-term ATR to long-term ATR to gauge whether volatility
    is expanding or contracting.

    Formula:
        ATR_Ratio = ATR(short_period) / ATR(long_period)

    Interpretation:
        < 1: consolidating (short-term vol below long-term)
        > 1: expanding (short-term vol exceeding long-term)

    Reference: Derivative of ATR analysis; conceptually related to
    the Choppiness Index but simpler and faster to compute.
    """
    n = len(closes)
    if n == 0 or n < long_period + 1:
        return np.full(n, np.nan)

    # True Range
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    def _wilder_atr(tr_arr, period):
        """Compute ATR with Wilder's smoothing."""
        length = len(tr_arr)
        atr_out = np.full(length, np.nan)
        atr_out[period] = np.mean(tr_arr[1 : period + 1])
        alpha = 1.0 / period
        for j in range(period + 1, length):
            atr_out[j] = atr_out[j - 1] * (1.0 - alpha) + tr_arr[j] * alpha
        return atr_out

    atr_short = _wilder_atr(tr, short_period)
    atr_long = _wilder_atr(tr, long_period)

    out = np.full(n, np.nan)
    for i in range(n):
        if (not np.isnan(atr_short[i]) and not np.isnan(atr_long[i])
                and atr_long[i] != 0):
            out[i] = atr_short[i] / atr_long[i]

    return out
