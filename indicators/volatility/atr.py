import numpy as np


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range using Wilder's smoothing (RMA).

    Returns an array of ATR values with the first `period` entries as np.nan.
    """
    n = len(closes)
    if n == 0 or n < period + 1:
        return np.full(n, np.nan)

    # True Range: max(H-L, |H - prev_close|, |L - prev_close|)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # Wilder's smoothing (RMA)
    out = np.full(n, np.nan)
    # Seed: simple average of first `period` true-range values (indices 1..period)
    out[period] = np.mean(tr[1 : period + 1])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        out[i] = out[i - 1] * (1.0 - alpha) + tr[i] * alpha

    return out
