import numpy as np


def natr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Normalized Average True Range (NATR).

    NATR = ATR / Close * 100

    Expresses ATR as a percentage of the closing price, making it comparable
    across instruments with different price levels.

    Reference: Widely used extension of Wilder's ATR (1978).
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

    # Wilder's smoothing (RMA) for ATR
    atr_vals = np.full(n, np.nan)
    atr_vals[period] = np.mean(tr[1 : period + 1])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr_vals[i] = atr_vals[i - 1] * (1.0 - alpha) + tr[i] * alpha

    # Normalize: ATR / Close * 100
    out = np.full(n, np.nan)
    valid = ~np.isnan(atr_vals) & (closes != 0)
    out[valid] = atr_vals[valid] / closes[valid] * 100.0

    return out
