"""Tests for volatility indicators: ATR, Bollinger Bands, Historical Volatility."""

import numpy as np
import pytest
from indicators.volatility.atr import atr
from indicators.volatility.bollinger import bollinger_bands, bollinger_width
from indicators.volatility.historical_vol import historical_volatility


# ── ATR ──────────────────────────────────────────────────────────────────────


def test_atr_basic():
    """Known input → verify ATR seed value is mean of first `period` true ranges."""
    period = 3
    highs = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
    lows = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    closes = np.array([11.0, 12.0, 13.0, 14.0, 15.0])

    result = atr(highs, lows, closes, period=period)

    # TR[0] = 12 - 10 = 2
    # TR[1] = max(13-11, |13-11|, |11-11|) = max(2, 2, 0) = 2
    # TR[2] = max(14-12, |14-12|, |12-12|) = max(2, 2, 0) = 2
    # TR[3] = max(15-13, |15-13|, |13-13|) = max(2, 2, 0) = 2
    # Seed at index 3: mean(TR[1:4]) = mean(2,2,2) = 2.0
    np.testing.assert_allclose(result[period], 2.0)
    assert len(result) == len(closes)


def test_atr_warmup():
    """First `period` values are np.nan."""
    period = 5
    n = 20
    highs = np.arange(n, dtype=float) + 2.0
    lows = np.arange(n, dtype=float)
    closes = np.arange(n, dtype=float) + 1.0

    result = atr(highs, lows, closes, period=period)
    assert all(np.isnan(result[:period]))
    assert not np.isnan(result[period])


def test_atr_constant_price():
    """Constant prices → ATR approaches 0."""
    period = 5
    n = 50
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    closes = np.full(n, 100.0)

    result = atr(highs, lows, closes, period=period)
    valid = result[~np.isnan(result)]
    np.testing.assert_allclose(valid, 0.0, atol=1e-10)


# ── Bollinger Bands ──────────────────────────────────────────────────────────


def test_bollinger_basic():
    """Middle band equals SMA."""
    np.random.seed(7)
    closes = np.cumsum(np.random.randn(60)) + 100.0
    period = 20
    upper, middle, lower = bollinger_bands(closes, period=period)

    # Manually compute SMA at a few indices and compare
    for i in [19, 30, 50]:
        expected_sma = np.mean(closes[i - period + 1 : i + 1])
        np.testing.assert_allclose(middle[i], expected_sma)


def test_bollinger_width():
    """Verify bollinger_width = (upper - lower) / middle."""
    np.random.seed(3)
    closes = np.cumsum(np.random.randn(60)) + 100.0
    period = 20

    upper, middle, lower = bollinger_bands(closes, period=period)
    width = bollinger_width(closes, period=period)

    valid = ~np.isnan(middle)
    expected = (upper[valid] - lower[valid]) / middle[valid]
    np.testing.assert_allclose(width[valid], expected)


def test_bollinger_symmetry():
    """Upper - middle == middle - lower (bands are symmetric)."""
    np.random.seed(11)
    closes = np.cumsum(np.random.randn(60)) + 100.0
    upper, middle, lower = bollinger_bands(closes, period=20)

    valid = ~np.isnan(middle)
    upper_diff = upper[valid] - middle[valid]
    lower_diff = middle[valid] - lower[valid]
    np.testing.assert_allclose(upper_diff, lower_diff)


# ── Historical Volatility ────────────────────────────────────────────────────


def test_histvol_basic():
    """Output should be positive where valid."""
    np.random.seed(1)
    closes = np.cumsum(np.random.randn(100)) + 100.0
    # Ensure all prices are positive
    closes = np.abs(closes) + 1.0

    result = historical_volatility(closes, period=20)
    valid = result[~np.isnan(result)]
    assert valid.size > 0
    assert np.all(valid > 0.0)


def test_histvol_constant():
    """Constant prices → volatility is 0 (or near 0)."""
    closes = np.full(50, 100.0)
    result = historical_volatility(closes, period=20)
    valid = result[~np.isnan(result)]
    np.testing.assert_allclose(valid, 0.0, atol=1e-10)


def test_histvol_warmup():
    """First `period` values are np.nan."""
    np.random.seed(5)
    closes = np.cumsum(np.random.randn(50)) + 100.0
    closes = np.abs(closes) + 1.0
    period = 20

    result = historical_volatility(closes, period=period)
    assert all(np.isnan(result[:period]))
    assert not np.isnan(result[period])
