import numpy as np


def vpt(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Volume-Price Trend (VPT).

    Cumulative indicator that weights volume by the fractional price change:
      VPT[i] = VPT[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1]

    Similar to OBV but proportional to the magnitude of price moves rather
    than just direction. First element is 0.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    n = len(closes)
    out = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if closes[i - 1] != 0:
            out[i] = out[i - 1] + volumes[i] * (closes[i] - closes[i - 1]) / closes[i - 1]
        else:
            out[i] = out[i - 1]

    return out
