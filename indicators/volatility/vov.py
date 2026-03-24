import numpy as np


def vov(
    closes: np.ndarray,
    vol_period: int = 20,
    vov_period: int = 20,
) -> np.ndarray:
    """Volatility of Volatility: std of rolling realized volatility.

    1. Compute rolling `vol_period`-bar std of log returns (realized vol).
    2. Compute rolling `vov_period`-bar std of that vol series.

    High VoV = unstable volatility regime.  Low VoV = stable regime.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    # Need vol_period+1 prices for vol_period returns, then vov_period vol values
    min_prices = vol_period + vov_period + 1
    if n < min_prices:
        return out

    safe = np.maximum(closes, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    # Step 1: rolling realized volatility
    vol_series_len = len(log_ret) - vol_period + 1
    if vol_series_len < vov_period:
        return out

    vol_series = np.empty(vol_series_len, dtype=np.float64)
    for i in range(vol_series_len):
        vol_series[i] = np.std(log_ret[i : i + vol_period], ddof=0)

    # Step 2: rolling std of vol series
    for i in range(vov_period, len(vol_series) + 1):
        window = vol_series[i - vov_period : i]
        # vol_series[i-1] uses log_ret up to index (i-1+vol_period-1),
        # which needs closes up to index (i-1+vol_period).
        closes_idx = (i - 1) + vol_period
        out[closes_idx] = np.std(window, ddof=0)

    return out
