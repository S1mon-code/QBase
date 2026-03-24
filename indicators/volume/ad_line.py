import numpy as np


def ad_line(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Accumulation/Distribution Line (Marc Chaikin).

    Cumulative indicator based on Close Location Value (CLV) weighted by volume:
      CLV = ((Close - Low) - (High - Close)) / (High - Low)
      AD[i] = AD[i-1] + CLV[i] * Volume[i]

    Unlike OBV which only considers direction, AD Line is proportional to where
    the close falls within the bar's range.

    Source: StockCharts / Marc Chaikin.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    hl_range = highs - lows

    # CLV = ((C - L) - (H - C)) / (H - L);  0 when H == L
    clv = np.where(hl_range != 0.0, ((closes - lows) - (highs - closes)) / hl_range, 0.0)

    money_flow_volume = clv * volumes

    return np.cumsum(money_flow_volume)
