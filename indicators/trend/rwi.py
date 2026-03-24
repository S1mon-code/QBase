import numpy as np


def rwi(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Random Walk Index (Michael Poulos, 1991).

    Compares price movement to a random walk to determine trend strength.

    For each bar, over look-back lengths k = 1 .. period:
      RWI_High_k = (High_t - Low_{t-k}) / (ATR_k * sqrt(k))
      RWI_Low_k  = (High_{t-k} - Low_t) / (ATR_k * sqrt(k))

    where ATR_k is the average true range over the k bars.  The final
    RWI High / Low is the maximum across all k.

    Values > 1.0 indicate a statistically significant trend.  RWI High > 1
    suggests an uptrend; RWI Low > 1 suggests a downtrend.

    Returns (rwi_high, rwi_low). First *period* values are np.nan.
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    rwi_high = np.full(n, np.nan, dtype=np.float64)
    rwi_low = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return rwi_high, rwi_low

    # Pre-compute true range (starts at index 1)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    for i in range(period, n):
        max_rwi_h = 0.0
        max_rwi_l = 0.0

        for k in range(1, period + 1):
            # Average true range over k bars ending at i
            atr_k = np.mean(tr[i - k + 1 : i + 1])
            denom = atr_k * np.sqrt(k)
            if denom <= 0.0:
                continue
            rwi_h = (highs[i] - lows[i - k]) / denom
            rwi_l = (highs[i - k] - lows[i]) / denom
            if rwi_h > max_rwi_h:
                max_rwi_h = rwi_h
            if rwi_l > max_rwi_l:
                max_rwi_l = rwi_l

        rwi_high[i] = max_rwi_h
        rwi_low[i] = max_rwi_l

    return rwi_high, rwi_low
