import numpy as np


def keltner(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channel.

    Middle line is an EMA of closes.  Upper and lower bands are
    middle +/- multiplier * ATR.

    Returns (upper, middle, lower).  First max(ema_period, atr_period)
    values are np.nan.
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # --- EMA ---
    alpha = 2.0 / (ema_period + 1)
    mid = np.empty(n, dtype=np.float64)
    mid[0] = closes[0]
    for i in range(1, n):
        mid[i] = alpha * closes[i] + (1.0 - alpha) * mid[i - 1]

    # --- True Range ---
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # --- ATR (simple rolling mean of last atr_period TR values) ---
    atr = np.full(n, np.nan, dtype=np.float64)
    if n >= atr_period:
        cumsum = np.cumsum(tr)
        atr[atr_period - 1] = cumsum[atr_period - 1] / atr_period
        for i in range(atr_period, n):
            atr[i] = (cumsum[i] - cumsum[i - atr_period]) / atr_period

    # --- Bands ---
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle_out = np.full(n, np.nan, dtype=np.float64)

    warmup = max(ema_period, atr_period)
    for i in range(warmup - 1, n):
        if np.isnan(atr[i]):
            continue
        middle_out[i] = mid[i]
        upper[i] = mid[i] + multiplier * atr[i]
        lower[i] = mid[i] - multiplier * atr[i]

    return upper, middle_out, lower
