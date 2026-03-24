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


def ppo(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentage Price Oscillator.

    PPO = (EMA_fast - EMA_slow) / EMA_slow * 100
    Signal = EMA(PPO, signal_period)
    Histogram = PPO - Signal

    Like MACD but percentage-based, enabling cross-security comparison.
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty, empty
    nans = np.full(n, np.nan)
    if n < slow:
        return nans.copy(), nans.copy(), nans.copy()

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)

    ppo_line = np.full(n, np.nan)
    valid = ~np.isnan(slow_ema) & (slow_ema != 0)
    ppo_line[valid] = (fast_ema[valid] - slow_ema[valid]) / slow_ema[valid] * 100.0

    # Signal line from valid PPO values
    valid_start = slow - 1
    ppo_valid = ppo_line[valid_start:]
    sig_ema = _ema(ppo_valid, signal)

    signal_line = np.full(n, np.nan)
    signal_line[valid_start:] = sig_ema

    histogram = ppo_line - signal_line

    return ppo_line, signal_line, histogram
