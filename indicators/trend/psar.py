import numpy as np


def psar(
    highs: np.ndarray,
    lows: np.ndarray,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic SAR (Welles Wilder).

    Algorithm:
      SAR_{t+1} = SAR_t + AF * (EP - SAR_t)
    where EP is the extreme point (highest high in uptrend / lowest low in
    downtrend) and AF starts at *af_start*, increments by *af_step* each time
    a new EP is recorded, capped at *af_max*.

    On a trend flip the SAR resets to the prior EP and AF resets to af_start.
    The SAR value is clamped so it never penetrates the prior two bars'
    range (high in downtrend, low in uptrend).

    Returns (psar_values, direction) where direction is 1 (bull) or -1 (bear).
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    sar = np.full(n, np.nan, dtype=np.float64)
    direction = np.ones(n, dtype=np.float64)

    if n < 2:
        sar[0] = lows[0]
        return sar, direction

    # Initialise: assume uptrend if second bar closes higher
    bull = highs[1] >= highs[0]
    af = af_start

    if bull:
        sar[0] = lows[0]
        ep = highs[0]
        direction[0] = 1.0
    else:
        sar[0] = highs[0]
        ep = lows[0]
        direction[0] = -1.0

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if bull:
            # Update EP
            if highs[i] > ep:
                ep = highs[i]
                af = min(af + af_step, af_max)

            # Calculate new SAR
            new_sar = prev_sar + af * (ep - prev_sar)

            # SAR must not be above the two prior lows
            new_sar = min(new_sar, lows[i - 1])
            if i >= 2:
                new_sar = min(new_sar, lows[i - 2])

            # Check for reversal
            if lows[i] < new_sar:
                bull = False
                new_sar = ep
                ep = lows[i]
                af = af_start
                direction[i] = -1.0
            else:
                direction[i] = 1.0
        else:
            # Update EP
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + af_step, af_max)

            # Calculate new SAR
            new_sar = prev_sar + af * (ep - prev_sar)

            # SAR must not be below the two prior highs
            new_sar = max(new_sar, highs[i - 1])
            if i >= 2:
                new_sar = max(new_sar, highs[i - 2])

            # Check for reversal
            if highs[i] > new_sar:
                bull = True
                new_sar = ep
                ep = highs[i]
                af = af_start
                direction[i] = 1.0
            else:
                direction[i] = -1.0

        sar[i] = new_sar

    return sar, direction
