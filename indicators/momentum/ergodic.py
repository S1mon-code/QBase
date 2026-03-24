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


def _ema_skip_nan(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA that skips leading NaNs, then applies standard EMA."""
    result = np.full(arr.size, np.nan)
    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return result
    first_valid = np.argmax(valid_mask)
    sub = arr[first_valid:]
    sub_ema = _ema(sub, period)
    result[first_valid:] = sub_ema
    return result


def ergodic(
    closes: np.ndarray,
    short_period: int = 5,
    long_period: int = 20,
    signal_period: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Ergodic Oscillator (William Blau).

    Double-smoothed momentum oscillator based on the True Strength Index.
    Ergodic = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)
    Signal = EMA(Ergodic, signal_period)

    Returns (ergodic_line, signal_line).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    nans = np.full(n, np.nan)
    if n <= long_period + short_period:
        return nans.copy(), nans.copy()

    momentum = np.diff(closes)
    abs_momentum = np.abs(momentum)

    # Double smooth: EMA(long) then EMA(short)
    ema1_mom = _ema(momentum, long_period)
    ema2_mom = _ema_skip_nan(ema1_mom, short_period)

    ema1_abs = _ema(abs_momentum, long_period)
    ema2_abs = _ema_skip_nan(ema1_abs, short_period)

    # Ergodic line (momentum array is n-1, indices shifted by 1)
    ergo_raw = np.full(n - 1, np.nan)
    valid = (~np.isnan(ema2_mom)) & (~np.isnan(ema2_abs)) & (ema2_abs != 0)
    ergo_raw[valid] = 100.0 * ema2_mom[valid] / ema2_abs[valid]

    ergo_line = np.full(n, np.nan)
    ergo_line[1:] = ergo_raw

    # Signal line
    sig_raw = _ema_skip_nan(ergo_raw, signal_period)
    signal_line = np.full(n, np.nan)
    signal_line[1:] = sig_raw

    return ergo_line, signal_line
