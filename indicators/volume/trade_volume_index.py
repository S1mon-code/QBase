import numpy as np


def tvi(
    closes: np.ndarray,
    volumes: np.ndarray,
    min_tick: float = 0.5,
) -> np.ndarray:
    """Trade Volume Index (TVI).

    Cumulative volume indicator that classifies each bar's volume as
    accumulation or distribution based on the price change relative to
    a minimum tick threshold:

      direction starts as ACCUMULATE
      if (close - prev_close) >  min_tick: direction = ACCUMULATE
      if (close - prev_close) < -min_tick: direction = DISTRIBUTE
      (otherwise direction unchanged)

      TVI[i] = TVI[i-1] + direction * volume[i]

    First value is 0.

    Source: General technical analysis (William Aspray adaptation).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.zeros(n, dtype=np.float64)
    direction = 1.0  # start as accumulation

    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        if change > min_tick:
            direction = 1.0
        elif change < -min_tick:
            direction = -1.0
        # else: direction unchanged

        result[i] = result[i - 1] + direction * volumes[i]

    return result
