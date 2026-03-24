import numpy as np


def fractal_high(highs: np.ndarray, period: int = 2) -> np.ndarray:
    """Williams Fractal — highs.

    A fractal high at bar i is True when highs[i] is strictly greater than
    all `period` bars on each side.  The first and last `period` bars are
    always False (insufficient context).

    Returns a boolean np.ndarray of the same length as `highs`.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=bool)

    out = np.zeros(n, dtype=bool)
    for i in range(period, n - period):
        is_fractal = True
        for offset in range(1, period + 1):
            if highs[i - offset] >= highs[i] or highs[i + offset] >= highs[i]:
                is_fractal = False
                break
        out[i] = is_fractal
    return out


def fractal_low(lows: np.ndarray, period: int = 2) -> np.ndarray:
    """Williams Fractal — lows.

    A fractal low at bar i is True when lows[i] is strictly less than
    all `period` bars on each side.

    Returns a boolean np.ndarray of the same length as `lows`.
    """
    n = len(lows)
    if n == 0:
        return np.array([], dtype=bool)

    out = np.zeros(n, dtype=bool)
    for i in range(period, n - period):
        is_fractal = True
        for offset in range(1, period + 1):
            if lows[i - offset] <= lows[i] or lows[i + offset] <= lows[i]:
                is_fractal = False
                break
        out[i] = is_fractal
    return out


def fractal_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Last confirmed fractal price levels.

    Returns (last_fractal_high, last_fractal_low) — two arrays where each
    element holds the price of the most recently confirmed fractal high/low
    at that bar.  Values are np.nan until the first fractal appears.
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    fh = fractal_high(highs, period)
    fl = fractal_low(lows, period)

    last_high = np.full(n, np.nan, dtype=np.float64)
    last_low = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if fh[i]:
            last_high[i] = highs[i]
        elif i > 0:
            last_high[i] = last_high[i - 1]

        if fl[i]:
            last_low[i] = lows[i]
        elif i > 0:
            last_low[i] = last_low[i - 1]

    return last_high, last_low
