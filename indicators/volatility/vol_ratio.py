import numpy as np


def volatility_ratio(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Volatility Ratio (Jack Schwager).

    Compares today's True Range to the ATR over a lookback period.
    Values > 1 suggest a potential breakout; values < 1 indicate
    contraction.

    Formula:
        VR = True_Range / ATR(period)

    Reference: Schwager, J.D. (1996), "Technical Analysis."
    """
    n = len(closes)
    if n == 0 or n < period + 1:
        return np.full(n, np.nan)

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

    # Volatility Ratio = TR / ATR
    out = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(atr_vals[i]) and atr_vals[i] != 0:
            out[i] = tr[i] / atr_vals[i]

    return out
