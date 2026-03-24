import numpy as np


def trend_intensity(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Trend Intensity Index (TII).

    Measures the proportion of price deviations above vs below a simple
    moving average over a lookback window, expressed as a percentage.

    Calculation (using half-period SMA as the reference, following the
    standard Dorsey formulation):
      1. Compute SMA over 2*period bars.
      2. For the most recent *period* closes, compute deviations from SMA:
         - SD+ = sum of (close - SMA) for closes above SMA
         - SD- = sum of (SMA - close) for closes below SMA
      3. TII = 100 * SD+ / (SD+ + SD-)

    Range is [0, 100].  Values > 50 indicate uptrend, < 50 indicate downtrend.
    First (2*period - 1) values are np.nan.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    sma_period = 2 * period
    out = np.full(n, np.nan, dtype=np.float64)

    if n < sma_period:
        return out

    # Compute SMA of length 2*period using cumsum
    cumsum = np.cumsum(closes)
    sma = np.full(n, np.nan, dtype=np.float64)
    sma[sma_period - 1] = cumsum[sma_period - 1] / sma_period
    if n > sma_period:
        sma[sma_period:] = (cumsum[sma_period:] - cumsum[:-sma_period]) / sma_period

    # TII requires *period* bars after the SMA becomes valid
    start = sma_period + period - 1
    if start >= n:
        # Edge case: check if we can compute at least one value
        # We need sma valid at index i and period bars of closes ending at i
        return out

    for i in range(sma_period - 1, n):
        # Look back *period* bars from i (inclusive)
        lb_start = i - period + 1
        if lb_start < 0:
            continue
        sma_val = sma[i]
        if np.isnan(sma_val):
            continue

        sd_pos = 0.0
        sd_neg = 0.0
        for j in range(lb_start, i + 1):
            diff = closes[j] - sma_val
            if diff > 0:
                sd_pos += diff
            else:
                sd_neg += -diff

        total = sd_pos + sd_neg
        if total == 0.0:
            out[i] = 50.0  # price equals SMA exactly
        else:
            out[i] = 100.0 * sd_pos / total

    return out
