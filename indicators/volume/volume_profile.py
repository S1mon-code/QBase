import numpy as np


def volume_profile(
    closes: np.ndarray,
    volumes: np.ndarray,
    bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Volume at Price (simplified volume profile).

    Distributes total volume into ``bins`` equal-width price buckets across
    the full price range. Returns:
      (price_levels, volume_at_level)

    ``price_levels`` are the midpoints of each bin.

    Returns empty arrays if input is empty.

    Source: General market profile / volume profile concept.
    """
    if closes.size == 0:
        return (np.array([], dtype=np.float64), np.array([], dtype=np.float64))

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    lo = np.min(closes)
    hi = np.max(closes)

    if lo == hi:
        # All closes at same price
        return (np.array([lo], dtype=np.float64), np.array([np.sum(volumes)], dtype=np.float64))

    # Create bin edges
    edges = np.linspace(lo, hi, bins + 1)
    vol_at_level = np.zeros(bins, dtype=np.float64)

    for i in range(len(closes)):
        # Determine which bin this close falls into
        idx = int((closes[i] - lo) / (hi - lo) * bins)
        if idx >= bins:
            idx = bins - 1
        vol_at_level[idx] += volumes[i]

    # Midpoints
    price_levels = (edges[:-1] + edges[1:]) / 2.0

    return (price_levels, vol_at_level)


def poc(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
    bins: int = 20,
) -> np.ndarray:
    """Point of Control — rolling price level with the highest volume.

    For each bar from ``period - 1`` onward, computes a volume profile over
    the trailing ``period`` bars and returns the price level (bin midpoint)
    with the most volume. Earlier values are np.nan.

    Source: Market Profile / volume profile concept.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        window_c = closes[i - period + 1 : i + 1]
        window_v = volumes[i - period + 1 : i + 1]

        lo = np.min(window_c)
        hi = np.max(window_c)

        if lo == hi:
            result[i] = lo
            continue

        edges = np.linspace(lo, hi, bins + 1)
        vol_at = np.zeros(bins, dtype=np.float64)

        for j in range(len(window_c)):
            idx = int((window_c[j] - lo) / (hi - lo) * bins)
            if idx >= bins:
                idx = bins - 1
            vol_at[idx] += window_v[j]

        best = int(np.argmax(vol_at))
        result[i] = (edges[best] + edges[best + 1]) / 2.0

    return result
