import numpy as np


def hurst_exponent(
    data: np.ndarray,
    max_lag: int = 20,
) -> np.ndarray:
    """Rolling Hurst exponent via rescaled range (R/S) method.

    Computed on a rolling window of `max_lag` returns using chunk sizes
    [8, 16, min(n//2, 32)] for the R/S regression.

    H > 0.5 = trending, H < 0.5 = mean reverting, H = 0.5 = random walk.
    Input `data` is a price series; log returns are computed internally.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Need at least 2 prices to get 1 return
    if n < 2:
        return np.full(n, np.nan, dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    # Log returns
    safe = np.maximum(data, 1e-9)
    log_prices = np.log(safe)
    log_ret = np.diff(log_prices)  # length n-1

    # We need at least max_lag returns for one window
    if len(log_ret) < max_lag:
        return out

    for i in range(max_lag, len(log_ret) + 1):
        ts = log_ret[i - max_lag : i]
        out[i] = _hurst_rs(ts)

    return out


def _hurst_rs(ts: np.ndarray) -> float:
    """Rescaled-range Hurst exponent estimate for a single window."""
    n = len(ts)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 32)
    sizes = []
    rs_vals = []

    for k in [8, 16, max_k]:
        if k < 8 or k > n:
            continue
        num_chunks = n // k
        if num_chunks < 1:
            continue
        rs_list = []
        for j in range(num_chunks):
            chunk = ts[j * k : (j + 1) * k]
            m = np.mean(chunk)
            s = np.std(chunk, ddof=0)
            if s < 1e-12:
                continue
            y = np.cumsum(chunk - m)
            r = np.max(y) - np.min(y)
            rs_list.append(r / s)
        if len(rs_list) > 0:
            sizes.append(k)
            rs_vals.append(np.mean(rs_list))

    if len(sizes) < 2:
        return 0.5

    log_s = np.log(np.array(sizes, dtype=np.float64))
    log_rs = np.log(np.array(rs_vals, dtype=np.float64))
    slope = np.polyfit(log_s, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))
