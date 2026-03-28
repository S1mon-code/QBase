"""Tests for alpha decay detection module."""
import sys
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pytest

from attribution.decay import (
    compute_rolling_ic,
    detect_decay_alert,
    _load_indicator_func,
    _INDICATOR_MAP,
)


# =========================================================================
# compute_rolling_ic
# =========================================================================

class TestComputeRollingIC:
    """Tests for rolling Information Coefficient computation."""

    def test_positively_correlated_data(self):
        """IC should be positive for positively correlated indicator and returns."""
        np.random.seed(42)
        n = 500
        # Indicator that predicts returns: returns = 0.3 * indicator + noise
        indicator = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        returns = 0.3 * indicator + noise

        ic = compute_rolling_ic(indicator, returns, window=100)

        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0, "Should have valid IC values"
        mean_ic = float(np.mean(valid))
        assert mean_ic > 0.1, f"Mean IC should be positive for correlated data, got {mean_ic:.4f}"

    def test_uncorrelated_data(self):
        """IC should be approximately zero for uncorrelated data."""
        np.random.seed(123)
        n = 1000
        indicator = np.random.randn(n)
        returns = np.random.randn(n)

        ic = compute_rolling_ic(indicator, returns, window=100)

        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0
        mean_ic = float(np.mean(valid))
        assert abs(mean_ic) < 0.1, f"Mean IC should be ~0 for uncorrelated data, got {mean_ic:.4f}"

    def test_negatively_correlated_data(self):
        """IC should be negative for negatively correlated data."""
        np.random.seed(42)
        n = 500
        indicator = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        returns = -0.3 * indicator + noise

        ic = compute_rolling_ic(indicator, returns, window=100)

        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0
        mean_ic = float(np.mean(valid))
        assert mean_ic < -0.1, f"Mean IC should be negative, got {mean_ic:.4f}"

    def test_short_data_returns_nan(self):
        """Data shorter than window should return all NaN."""
        indicator = np.array([1.0, 2.0, 3.0])
        returns = np.array([0.01, 0.02, 0.03])

        ic = compute_rolling_ic(indicator, returns, window=100)
        assert np.all(np.isnan(ic))

    def test_nan_handling(self):
        """Should handle NaN values in input gracefully."""
        np.random.seed(42)
        n = 300
        indicator = np.random.randn(n)
        returns = 0.3 * indicator + np.random.randn(n) * 0.5

        # Inject NaNs
        indicator[50:60] = np.nan
        returns[100:110] = np.nan

        ic = compute_rolling_ic(indicator, returns, window=100)

        # Should still produce valid ICs (just skip NaN bars in windows)
        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0

    def test_ic_range(self):
        """IC values should be in [-1, 1] range."""
        np.random.seed(42)
        n = 500
        indicator = np.random.randn(n)
        returns = 0.5 * indicator + np.random.randn(n)

        ic = compute_rolling_ic(indicator, returns, window=100)
        valid = ic[~np.isnan(ic)]

        assert np.all(valid >= -1.0), "IC should be >= -1"
        assert np.all(valid <= 1.0), "IC should be <= 1"


# =========================================================================
# detect_decay_alert
# =========================================================================

class TestDetectDecayAlert:
    """Tests for the decay alert detection function."""

    def test_no_alert_when_healthy(self):
        """No alert should fire when IC is consistently above threshold."""
        yearly_ics = {2020: 0.15, 2021: 0.12, 2022: 0.18, 2023: 0.10, 2024: 0.14}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 0

    def test_alert_on_two_consecutive_weak_years(self):
        """Alert should fire when IC < threshold for 2+ consecutive years."""
        yearly_ics = {2020: 0.15, 2021: 0.03, 2022: 0.02, 2023: 0.12, 2024: 0.10}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 1
        assert "2 consecutive years" in alerts[0]

    def test_no_alert_on_single_weak_year(self):
        """A single weak year should NOT trigger an alert."""
        yearly_ics = {2020: 0.15, 2021: 0.03, 2022: 0.12, 2023: 0.10}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 0

    def test_alert_on_trailing_weak_years(self):
        """Alert should fire when the most recent years are weak."""
        yearly_ics = {2020: 0.15, 2021: 0.12, 2022: 0.03, 2023: 0.02, 2024: 0.01}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 1
        assert "most recent" in alerts[0]

    def test_multiple_decay_periods(self):
        """Multiple decay periods should each generate an alert."""
        yearly_ics = {
            2018: 0.02, 2019: 0.01,  # decay period 1
            2020: 0.15,              # recovery
            2021: 0.03, 2022: 0.02,  # decay period 2
            2023: 0.15,              # recovery
        }
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 2

    def test_nan_treated_as_weak(self):
        """NaN IC values should be treated as below threshold."""
        yearly_ics = {2020: 0.15, 2021: np.nan, 2022: np.nan, 2023: 0.12}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 1

    def test_empty_input(self):
        """Empty input should return no alerts."""
        alerts = detect_decay_alert({}, threshold=0.05)
        assert len(alerts) == 0

    def test_all_years_strong(self):
        """All years strong -> no alerts."""
        yearly_ics = {yr: 0.20 for yr in range(2015, 2025)}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 0

    def test_all_years_weak(self):
        """All years weak -> one alert covering entire period."""
        yearly_ics = {yr: 0.01 for yr in range(2015, 2025)}
        alerts = detect_decay_alert(yearly_ics, threshold=0.05)
        assert len(alerts) == 1
        assert "10 consecutive years" in alerts[0]


# =========================================================================
# Indicator name -> function mapping
# =========================================================================

class TestIndicatorMapping:
    """Tests for indicator name to function mapping."""

    def test_known_indicator_loads(self):
        """Known indicators in the map should load successfully."""
        for name in ("volume_momentum", "rsi", "adx", "roc"):
            if name not in _INDICATOR_MAP:
                continue
            func, arrays = _load_indicator_func(name)
            assert callable(func), f"{name} should return a callable"
            assert arrays in ("c", "v", "cv", "hl", "hlc", "hlcv")

    def test_unknown_indicator_raises(self):
        """Unknown indicator name should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            _load_indicator_func("totally_fake_indicator_xyz")

    def test_volume_momentum_returns_array(self):
        """volume_momentum should return a numpy array when called."""
        func, arrays = _load_indicator_func("volume_momentum")
        assert arrays == "v"

        volumes = np.random.rand(100) * 10000 + 1000
        result = func(volumes)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(volumes)

    def test_indicator_map_completeness(self):
        """The indicator map should have entries for key indicators."""
        expected = {"volume_momentum", "rsi", "adx", "roc", "cci", "macd"}
        actual = set(_INDICATOR_MAP.keys())
        missing = expected - actual
        assert len(missing) == 0, f"Missing from indicator map: {missing}"
