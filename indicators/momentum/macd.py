import numpy as np


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. First `period-1` values are np.nan."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result

    result[period - 1] = arr[:period].mean()
    k = 2.0 / (period + 1.0)
    for i in range(period, arr.size):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence. Returns (macd_line, signal_line, histogram)."""
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty, empty

    nans = np.full(n, np.nan)
    if n < slow:
        return nans.copy(), nans.copy(), nans.copy()

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)

    macd_line = fast_ema - slow_ema

    # Signal line is EMA of the valid portion of the MACD line
    valid_start = slow - 1
    macd_valid = macd_line[valid_start:]
    sig_ema = _ema(macd_valid, signal)

    signal_line = np.full(n, np.nan)
    signal_line[valid_start:] = sig_ema

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
