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


def schaff_trend_cycle(
    closes: np.ndarray,
    period: int = 10,
    fast: int = 23,
    slow: int = 50,
) -> np.ndarray:
    """Schaff Trend Cycle.

    MACD run through a double stochastic smoothing:
    1. MACD = EMA(fast) - EMA(slow)
    2. %K1 = Stochastic of MACD over `period`
    3. %D1 = EMA(%K1, factor)  — first smoothing (factor = 0.5)
    4. %K2 = Stochastic of %D1 over `period`
    5. STC = EMA(%K2, factor) — second smoothing

    Range [0, 100].
    """
    n = closes.size
    if n == 0:
        return np.array([], dtype=float)
    if n < slow + period:
        return np.full(n, np.nan)

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = fast_ema - slow_ema

    factor = 0.5  # EMA smoothing constant for stochastic steps

    # First stochastic + EMA smoothing
    pf = np.full(n, np.nan)  # %D1 (PF in Schaff's notation)
    for i in range(slow - 1 + period - 1, n):
        window = macd_line[i - period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        lo = np.min(window)
        hi = np.max(window)
        if hi == lo:
            stoch = 50.0
        else:
            stoch = 100.0 * (macd_line[i] - lo) / (hi - lo)

        if np.isnan(pf[i - 1]):
            pf[i] = stoch
        else:
            pf[i] = pf[i - 1] + factor * (stoch - pf[i - 1])

    # Second stochastic + EMA smoothing
    stc = np.full(n, np.nan)
    for i in range(slow - 1 + 2 * (period - 1), n):
        window = pf[i - period + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) < period:
            continue
        lo = np.min(valid_window)
        hi = np.max(valid_window)
        if hi == lo:
            stoch = 50.0
        else:
            stoch = 100.0 * (pf[i] - lo) / (hi - lo)

        if np.isnan(stc[i - 1]):
            stc[i] = stoch
        else:
            stc[i] = stc[i - 1] + factor * (stoch - stc[i - 1])

    return stc
