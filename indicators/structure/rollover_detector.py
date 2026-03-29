import numpy as np


def rollover_detector(
    volumes: np.ndarray,
    factors: np.ndarray,
    vol_threshold: float = 0.18,
    factor_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect contract rollover events from volume spikes and adjustment factor changes.

    Two detection methods (OR logic):
    1. Volume change: |vol[i] - vol[i-1]| / vol[i-1] > vol_threshold
    2. Factor change: |factor[i] - factor[i-1]| > factor_eps

    Returns (is_rollover, vol_change):
      is_rollover — boolean array, True on detected rollover bars
      vol_change  — absolute percentage volume change at each bar
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    volumes = volumes.astype(np.float64)
    factors = factors.astype(np.float64)

    is_rollover = np.zeros(n, dtype=bool)
    vol_change = np.full(n, np.nan, dtype=np.float64)

    vol_change[0] = 0.0
    for i in range(1, n):
        prev_vol = max(volumes[i - 1], 1.0)
        vc = abs(volumes[i] - volumes[i - 1]) / prev_vol
        vol_change[i] = vc

        vol_spike = vc > vol_threshold
        factor_changed = abs(factors[i] - factors[i - 1]) > factor_eps

        is_rollover[i] = vol_spike or factor_changed

    return is_rollover, vol_change


def post_rollover_momentum(
    closes: np.ndarray,
    is_rollover: np.ndarray,
    settle_bars: int = 4,
    mom_period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-rollover momentum signal.

    After a detected rollover, waits ``settle_bars`` for price to stabilize,
    then measures momentum over ``mom_period`` bars.

    Returns (post_roll_return, post_roll_mom):
      post_roll_return — return from rollover price to current close
      post_roll_mom    — mom_period-bar momentum at signal time
    """
    n = len(closes)
    post_roll_return = np.full(n, np.nan, dtype=np.float64)
    post_roll_mom = np.full(n, np.nan, dtype=np.float64)

    if n < mom_period + settle_bars:
        return post_roll_return, post_roll_mom

    closes = closes.astype(np.float64)
    is_roll = is_rollover.astype(bool)

    last_roll_price = np.nan
    bars_since_roll = -1

    for i in range(n):
        if is_roll[i]:
            last_roll_price = closes[i]
            bars_since_roll = 0
        elif bars_since_roll >= 0:
            bars_since_roll += 1

        if bars_since_roll == settle_bars and not np.isnan(last_roll_price):
            if last_roll_price > 1e-9:
                post_roll_return[i] = (closes[i] - last_roll_price) / last_roll_price

            if i >= mom_period:
                prev_close = closes[i - mom_period]
                if prev_close > 1e-9:
                    post_roll_mom[i] = closes[i] / prev_close - 1.0

    return post_roll_return, post_roll_mom
