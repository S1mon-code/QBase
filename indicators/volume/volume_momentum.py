import numpy as np


def volume_momentum(
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Volume Momentum.

    Ratio of current volume to its SMA, smoothed with an EMA:
      raw_ratio[i] = Volume[i] / SMA(Volume, period)[i]
      VM = EMA(raw_ratio, period)

    First ``2 * period - 2`` values are np.nan (period-1 for SMA, then
    period-1 for EMA).

    Source: General technical analysis reference.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=np.float64)

    volumes = volumes.astype(np.float64)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    # SMA of volume
    sma = np.full(n, np.nan, dtype=np.float64)
    running = np.sum(volumes[:period])
    sma[period - 1] = running / period
    for i in range(period, n):
        running += volumes[i] - volumes[i - period]
        sma[i] = running / period

    # Raw ratio: volume / SMA(volume)
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if sma[i] != 0.0:
            ratio[i] = volumes[i] / sma[i]
        else:
            ratio[i] = 0.0

    # EMA of ratio starting from first valid ratio
    valid_start = period - 1
    valid_ratios = ratio[valid_start:]
    if len(valid_ratios) < period:
        return result

    alpha = 2.0 / (period + 1)
    ema_val = np.mean(valid_ratios[:period])
    result[valid_start + period - 1] = ema_val

    for i in range(period, len(valid_ratios)):
        ema_val = alpha * valid_ratios[i] + (1.0 - alpha) * ema_val
        result[valid_start + i] = ema_val

    return result


def relative_volume(
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Relative Volume (RVOL).

    Simple ratio of current volume to its SMA:
      RVOL[i] = Volume[i] / SMA(Volume, period)[i]

    Values > 1 indicate above-average volume. First ``period - 1`` values
    are np.nan.

    Source: General technical analysis reference.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=np.float64)

    volumes = volumes.astype(np.float64)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    running = np.sum(volumes[:period])
    sma_val = running / period
    result[period - 1] = volumes[period - 1] / sma_val if sma_val != 0.0 else 0.0

    for i in range(period, n):
        running += volumes[i] - volumes[i - period]
        sma_val = running / period
        result[i] = volumes[i] / sma_val if sma_val != 0.0 else 0.0

    return result
