import numpy as np


def ttm_squeeze(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """TTM Squeeze: Bollinger Bands inside Keltner Channels.

    squeeze_on = True when BB upper < KC upper AND BB lower > KC lower.
    momentum   = linear regression slope of closes over bb_period.

    Returns (squeeze_on, momentum) arrays aligned to input length.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    warmup = max(bb_period, kc_period)
    squeeze_on = np.full(n, False, dtype=bool)
    momentum = np.full(n, np.nan, dtype=np.float64)

    # True Range for Keltner ATR (SMA-based)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    x = np.arange(bb_period, dtype=np.float64)

    for i in range(warmup - 1, n):
        # Bollinger Bands
        bb_slice = closes[i - bb_period + 1 : i + 1]
        bb_mid = np.mean(bb_slice)
        bb_std = np.std(bb_slice, ddof=0)
        bb_upper = bb_mid + bb_mult * bb_std
        bb_lower = bb_mid - bb_mult * bb_std

        # Keltner Channel (SMA + ATR * mult)
        kc_mid = np.mean(closes[i - kc_period + 1 : i + 1])
        kc_atr = np.mean(tr[i - kc_period + 1 : i + 1])
        kc_upper = kc_mid + kc_mult * kc_atr
        kc_lower = kc_mid - kc_mult * kc_atr

        squeeze_on[i] = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        # Momentum: linear regression slope over bb_period
        y = closes[i - bb_period + 1 : i + 1]
        coeffs = np.polyfit(x, y, 1)
        momentum[i] = coeffs[0]

    return squeeze_on, momentum
