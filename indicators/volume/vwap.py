import numpy as np


def vwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Volume Weighted Average Price (cumulative from start).

    VWAP = cumsum(TP * Volume) / cumsum(Volume).
    Returns full-length array.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    tp = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(tp * volumes)
    cum_vol = np.cumsum(volumes)

    # Avoid division by zero where cumulative volume is still 0
    result = np.where(cum_vol != 0.0, cum_tp_vol / cum_vol, np.nan)
    return result


def vwap_session(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    session_starts: np.ndarray,
) -> np.ndarray:
    """Session-resetting VWAP.

    Resets the cumulative VWAP calculation at each bar where
    session_starts is True.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)
    session_starts = np.asarray(session_starts, dtype=bool)

    tp = (highs + lows + closes) / 3.0
    tp_vol = tp * volumes

    n = len(closes)
    result = np.empty(n, dtype=np.float64)
    cum_tp_vol = 0.0
    cum_vol = 0.0

    for i in range(n):
        if session_starts[i]:
            cum_tp_vol = 0.0
            cum_vol = 0.0
        cum_tp_vol += tp_vol[i]
        cum_vol += volumes[i]
        result[i] = cum_tp_vol / cum_vol if cum_vol != 0.0 else np.nan

    return result
