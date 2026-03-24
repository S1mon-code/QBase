import numpy as np


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder's smoothing."""
    if closes.size <= period:
        return np.full(closes.size, np.nan)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(closes.size, np.nan)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    alpha = 1.0 / period
    for i in range(period, len(deltas)):
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return result


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """SMA over valid (non-NaN) portion."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result
    cumsum = np.nancumsum(arr)
    # Find first index where we have `period` consecutive valid values
    valid = ~np.isnan(arr)
    count = 0
    for i in range(arr.size):
        if valid[i]:
            count += 1
        else:
            count = 0
        if count >= period:
            start = i - period + 1
            result[i] = arr[start:i + 1].sum() / period

    return result


def stoch_rsi(
    closes: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI (Tushar Chande & Stanley Kroll).

    StochRSI = (RSI - RSI_low) / (RSI_high - RSI_low) over stoch_period
    %K = SMA(StochRSI, k_period)
    %D = SMA(%K, d_period)

    Range [0, 1] (or [0, 100] when multiplied by 100).
    Here we return values in [0, 1] scale.
    Returns (%K, %D).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    nans = np.full(n, np.nan)
    if n <= rsi_period + stoch_period:
        return nans.copy(), nans.copy()

    rsi_values = _rsi(closes, rsi_period)

    # Stochastic of RSI
    stoch_raw = np.full(n, np.nan)
    for i in range(rsi_period + stoch_period - 1, n):
        window = rsi_values[i - stoch_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        lo = np.min(window)
        hi = np.max(window)
        if hi == lo:
            stoch_raw[i] = 0.5
        else:
            stoch_raw[i] = (rsi_values[i] - lo) / (hi - lo)

    # %K = SMA of stoch_raw
    k_line = _sma(stoch_raw, k_period)

    # %D = SMA of %K
    d_line = _sma(k_line, d_period)

    return k_line, d_line
