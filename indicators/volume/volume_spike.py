import numpy as np


def volume_spike(
    volumes: np.ndarray, period: int = 20, threshold: float = 2.0
) -> np.ndarray:
    """Detect volume spikes.

    Returns a boolean array where True indicates the bar's volume exceeds
    ``threshold`` times the rolling ``period``-bar average volume.
    """
    if volumes.size == 0:
        return np.array([], dtype=bool)

    n = len(volumes)
    volumes = volumes.astype(np.float64)
    out = np.zeros(n, dtype=bool)

    for i in range(period, n):
        avg = np.mean(volumes[i - period:i])
        if avg > 0:
            out[i] = volumes[i] > threshold * avg

    return out


def volume_climax(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Volume Climax score.

    Identifies bars with exceptionally high volume AND wide range AND
    close near an extreme of the bar. Returns a signed score in roughly
    [-1, 1]: positive for bullish climax (close near high), negative for
    bearish climax (close near low). Near-zero when no climax.

    Score = volume_ratio * range_ratio * close_location, where:
      - volume_ratio = volume / avg_volume (capped contribution)
      - range_ratio  = bar_range / avg_range (capped contribution)
      - close_location = (close - low) / (high - low) mapped to [-1, 1]
    The product is normalised so typical values stay within [-1, 1].
    """
    if highs.size == 0:
        return np.array([], dtype=np.float64)

    n = len(highs)
    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    ranges = highs - lows

    for i in range(period, n):
        avg_vol = np.mean(volumes[i - period:i])
        avg_range = np.mean(ranges[i - period:i])

        if avg_vol <= 0 or avg_range <= 0 or ranges[i] <= 0:
            out[i] = 0.0
            continue

        # Ratios capped at 5x to avoid extreme outliers dominating
        vol_r = min(volumes[i] / avg_vol, 5.0) / 5.0   # [0, 1]
        rng_r = min(ranges[i] / avg_range, 5.0) / 5.0   # [0, 1]

        # Close location: 0 at low, 1 at high -> remap to [-1, 1]
        clv = 2.0 * (closes[i] - lows[i]) / ranges[i] - 1.0

        out[i] = vol_r * rng_r * clv

    return out


def volume_dry_up(
    volumes: np.ndarray, period: int = 20, threshold: float = 0.5
) -> np.ndarray:
    """Detect volume dry-up bars.

    Returns a boolean array where True indicates the bar's volume is below
    ``threshold`` times the rolling ``period``-bar average (selling/buying
    exhaustion).
    """
    if volumes.size == 0:
        return np.array([], dtype=bool)

    n = len(volumes)
    volumes = volumes.astype(np.float64)
    out = np.zeros(n, dtype=bool)

    for i in range(period, n):
        avg = np.mean(volumes[i - period:i])
        if avg > 0:
            out[i] = volumes[i] < threshold * avg

    return out
