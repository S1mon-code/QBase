import numpy as np


def stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               k_period: int = 14, d_period: int = 3
               ) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator. Returns (%K, %D) where %D is SMA of %K."""
    if closes.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = closes.size
    if n < k_period:
        return np.full(n, np.nan), np.full(n, np.nan)

    k = np.full(n, np.nan)

    for i in range(k_period - 1, n):
        highest = np.max(highs[i - k_period + 1:i + 1])
        lowest = np.min(lows[i - k_period + 1:i + 1])
        if highest == lowest:
            k[i] = 50.0
        else:
            k[i] = (closes[i] - lowest) / (highest - lowest) * 100.0

    # %D = simple moving average of %K
    d = np.full(n, np.nan)
    for i in range(k_period - 1 + d_period - 1, n):
        seg = k[i - d_period + 1:i + 1]
        if np.any(np.isnan(seg)):
            continue
        d[i] = np.mean(seg)

    return k, d


def kdj(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 9, k_smooth: int = 3, d_smooth: int = 3
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDJ indicator. Returns (K, D, J) where J = 3K - 2D.

    K and D use exponential-style smoothing:
        K = prev_K * (k_smooth-1)/k_smooth + RSV / k_smooth
        D = prev_D * (d_smooth-1)/d_smooth + K   / d_smooth
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    n = closes.size
    if n < period:
        nan_arr = np.full(n, np.nan)
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    k_out = np.full(n, np.nan)
    d_out = np.full(n, np.nan)
    j_out = np.full(n, np.nan)

    for i in range(period - 1, n):
        highest = np.max(highs[i - period + 1:i + 1])
        lowest = np.min(lows[i - period + 1:i + 1])
        if highest == lowest:
            rsv = 50.0
        else:
            rsv = (closes[i] - lowest) / (highest - lowest) * 100.0

        if i == period - 1:
            k_out[i] = rsv
            d_out[i] = rsv
        else:
            k_out[i] = k_out[i - 1] * (k_smooth - 1) / k_smooth + rsv / k_smooth
            d_out[i] = d_out[i - 1] * (d_smooth - 1) / d_smooth + k_out[i] / d_smooth

        j_out[i] = 3.0 * k_out[i] - 2.0 * d_out[i]

    return k_out, d_out, j_out
