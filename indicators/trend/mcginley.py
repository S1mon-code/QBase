import numpy as np


def mcginley_dynamic(data: np.ndarray, period: int = 14) -> np.ndarray:
    """McGinley Dynamic indicator.

    A self-adjusting moving average that adapts to market speed.

    Formula:
      MD_t = MD_{t-1} + (Price - MD_{t-1}) / (N * (Price / MD_{t-1})^4)

    where N is the smoothing period.  The (Price/MD)^4 term causes the
    indicator to accelerate when price moves away from it and decelerate
    when price is close, effectively eliminating whipsaws.

    Initialised with the first data value; no NaN warmup.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]

    for i in range(1, n):
        prev = out[i - 1]
        price = data[i]
        if prev == 0.0:
            out[i] = price
            continue
        ratio = price / prev
        out[i] = prev + (price - prev) / (period * ratio ** 4)

    return out
