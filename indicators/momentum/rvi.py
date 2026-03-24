import numpy as np


def rvi(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative Vigor Index.

    Compares close-open range to high-low range using symmetric weighted
    smoothing (weights 1,2,2,1 over 4 bars) then averaged over `period`.

    RVI = SMA(numerator, period) / SMA(denominator, period)
    Signal = symmetrically weighted MA of RVI (weights 1,2,2,1).

    Returns (rvi_line, signal_line).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    warmup = period + 3 + 3  # 4-bar smoothing + SMA + signal smoothing
    if n < warmup:
        nans = np.full(n, np.nan)
        return nans.copy(), nans.copy()

    co = closes - opens   # close - open
    hl = highs - lows     # high - low

    # Symmetric 4-bar weighted average: (a + 2b + 2c + d) / 6
    num = np.full(n, np.nan)
    den = np.full(n, np.nan)
    for i in range(3, n):
        num[i] = (co[i] + 2.0 * co[i - 1] + 2.0 * co[i - 2] + co[i - 3]) / 6.0
        den[i] = (hl[i] + 2.0 * hl[i - 1] + 2.0 * hl[i - 2] + hl[i - 3]) / 6.0

    # Rolling sum over `period` for numerator and denominator
    rvi_line = np.full(n, np.nan)
    start = 3 + period - 1
    for i in range(start, n):
        sum_num = num[i - period + 1:i + 1].sum()
        sum_den = den[i - period + 1:i + 1].sum()
        if sum_den == 0:
            rvi_line[i] = 0.0
        else:
            rvi_line[i] = sum_num / sum_den

    # Signal line: symmetric weighted MA of RVI (1,2,2,1)/6
    signal_line = np.full(n, np.nan)
    for i in range(start + 3, n):
        signal_line[i] = (
            rvi_line[i] + 2.0 * rvi_line[i - 1]
            + 2.0 * rvi_line[i - 2] + rvi_line[i - 3]
        ) / 6.0

    return rvi_line, signal_line
