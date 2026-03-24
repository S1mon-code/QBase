import numpy as np


def vroc(volumes: np.ndarray, period: int = 14) -> np.ndarray:
    """Volume Rate of Change.

    Percentage change in volume relative to ``period`` bars ago:
      VROC = (Volume - Volume[i - period]) / Volume[i - period] * 100

    First ``period`` values are np.nan.

    Source: General technical analysis reference.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=np.float64)

    volumes = volumes.astype(np.float64)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        prev = volumes[i - period]
        if prev != 0.0:
            result[i] = (volumes[i] - prev) / prev * 100.0
        else:
            result[i] = 0.0

    return result
