import numpy as np


def alma(
    data: np.ndarray,
    period: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> np.ndarray:
    """Arnaud Legoux Moving Average (ALMA).

    A Gaussian-weighted moving average whose peak position is controlled by
    *offset* (0..1, where 0.85 puts 85 % of the weight toward recent data)
    and whose width is controlled by *sigma*.

    Weight calculation:
      m = offset * (period - 1)          # peak of gaussian
      s = period / sigma                  # std-dev of gaussian
      w[i] = exp(-((i - m) ** 2) / (2 * s ** 2))   for i = 0 .. period-1

    ALMA = sum(w * window) / sum(w)

    First period-1 values are np.nan.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    m = offset * (period - 1)
    s = period / sigma

    # Pre-compute normalised weights
    idx = np.arange(period, dtype=np.float64)
    w = np.exp(-((idx - m) ** 2) / (2.0 * s * s))
    w_sum = w.sum()

    for i in range(period - 1, n):
        out[i] = np.dot(data[i - period + 1 : i + 1], w) / w_sum

    return out
