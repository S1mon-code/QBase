import numpy as np


def nvi(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Negative Volume Index.

    Updates only on days when volume *decreases* from the prior day,
    reflecting "smart money" activity:
      if V[i] < V[i-1]: NVI[i] = NVI[i-1] + (C[i] - C[i-1]) / C[i-1] * NVI[i-1]
      else:              NVI[i] = NVI[i-1]

    Starts at 1000.

    Source: Paul Dysart (1936), popularised by Norman Fosback.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[0] = 1000.0

    for i in range(1, n):
        if volumes[i] < volumes[i - 1] and closes[i - 1] != 0.0:
            result[i] = result[i - 1] + (closes[i] - closes[i - 1]) / closes[i - 1] * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result


def pvi(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Positive Volume Index.

    Updates only on days when volume *increases* from the prior day,
    reflecting "crowd" activity:
      if V[i] > V[i-1]: PVI[i] = PVI[i-1] + (C[i] - C[i-1]) / C[i-1] * PVI[i-1]
      else:              PVI[i] = PVI[i-1]

    Starts at 1000.

    Source: Paul Dysart (1936), popularised by Norman Fosback.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[0] = 1000.0

    for i in range(1, n):
        if volumes[i] > volumes[i - 1] and closes[i - 1] != 0.0:
            result[i] = result[i - 1] + (closes[i] - closes[i - 1]) / closes[i - 1] * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result
