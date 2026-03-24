import numpy as np


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume.

    Cumulative volume indicator where volume is added on up-closes
    and subtracted on down-closes.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    direction = np.zeros(len(closes), dtype=np.float64)
    direction[1:] = np.sign(np.diff(closes))

    directed_volume = direction * volumes
    directed_volume[0] = volumes[0]

    return np.cumsum(directed_volume)
