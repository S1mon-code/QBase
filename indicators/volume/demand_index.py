import numpy as np


def demand_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Demand Index (James Sibbet).

    Combines price momentum and volume to gauge buying vs selling pressure.
    The calculation builds a ratio of buying pressure (BP) to selling
    pressure (SP):

      Price Change = (Close - prev_Close) / prev_Close * 100
      Volume Factor = Volume / SMA(Volume, period)

      BP = max(Price Change, 0) * Volume Factor + some base
      SP = max(-Price Change, 0) * Volume Factor + some base

      if BP >= SP:  DI =  100 * (1 - SP / BP)    (range 0 to +100)
      if SP >  BP:  DI = -100 * (1 - BP / SP)    (range -100 to 0)

    This simplified formulation captures Sibbet's core concept: positive
    when buying pressure dominates and negative when selling pressure
    dominates, scaled by relative volume.

    First ``period`` values are np.nan.

    Source: James Sibbet / Sierra Chart reference.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    if n <= period:
        return result

    # SMA of volume (rolling)
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    running = np.sum(volumes[:period])
    vol_sma[period - 1] = running / period
    for i in range(period, n):
        running += volumes[i] - volumes[i - period]
        vol_sma[i] = running / period

    # Compute Demand Index from bar `period` onward
    for i in range(period, n):
        if closes[i - 1] == 0.0 or vol_sma[i] == 0.0:
            result[i] = 0.0
            continue

        # Price change percentage
        pct_change = (closes[i] - closes[i - 1]) / closes[i - 1] * 100.0

        # Volume factor: how current volume compares to average
        vol_factor = volumes[i] / vol_sma[i]

        # Buying and selling pressure
        # Base component ensures neither BP nor SP is zero
        base = 1.0
        bp = max(pct_change, 0.0) * vol_factor + base
        sp = max(-pct_change, 0.0) * vol_factor + base

        if bp >= sp:
            result[i] = 100.0 * (1.0 - sp / bp) if bp != 0.0 else 0.0
        else:
            result[i] = -100.0 * (1.0 - bp / sp) if sp != 0.0 else 0.0

    return result
