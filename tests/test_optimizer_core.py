"""Tests for strategies/optimizer_core.py — narrow_param_space and optimize_two_phase."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from strategies.optimizer_core import narrow_param_space, optimize_two_phase


# =====================================================================
# narrow_param_space tests
# =====================================================================

def _make_spec(name, low, high, step=0.1, dtype="float"):
    return {"name": name, "low": low, "high": high, "step": step, "dtype": dtype}


class TestNarrowParamSpace:
    """Tests for narrow_param_space boundary protection."""

    def test_center_value_normal_shrink(self):
        """Value at center of range: normal ±shrink_ratio/2 narrowing."""
        specs = [_make_spec("x", 0.0, 10.0)]
        best = {"x": 5.0}
        result = narrow_param_space(specs, best, shrink_ratio=0.3)
        r = result[0]
        # half_range = 10 * 0.3 / 2 = 1.5, so [3.5, 6.5]
        assert abs(r["low"] - 3.5) < 1e-9
        assert abs(r["high"] - 6.5) < 1e-9
        # step halved
        assert abs(r["step"] - 0.05) < 1e-9

    def test_low_boundary_expands_range(self):
        """Value near low boundary: should expand to [orig_low, orig_low + 40% range]."""
        specs = [_make_spec("x", 0.0, 10.0)]
        # val=1.0 is within 20% of low boundary (0.0): 1.0 < 10.0 * 0.2 = 2.0
        best = {"x": 1.0}
        result = narrow_param_space(specs, best, shrink_ratio=0.3)
        r = result[0]
        assert abs(r["low"] - 0.0) < 1e-9
        assert abs(r["high"] - 4.0) < 1e-9  # orig_low + 0.4 * 10 = 4.0

    def test_high_boundary_expands_range(self):
        """Value near high boundary: should expand to [orig_high - 40% range, orig_high]."""
        specs = [_make_spec("x", 0.0, 10.0)]
        # val=9.5 is within 20% of high boundary: (10-9.5)=0.5 < 10*0.2=2.0
        best = {"x": 9.5}
        result = narrow_param_space(specs, best, shrink_ratio=0.3)
        r = result[0]
        assert abs(r["low"] - 6.0) < 1e-9   # orig_high - 0.4 * 10 = 6.0
        assert abs(r["high"] - 10.0) < 1e-9

    def test_exact_boundary_value(self):
        """Value exactly at the boundary edge."""
        specs = [_make_spec("atr_stop_mult", 2.0, 15.0)]
        best = {"atr_stop_mult": 14.29}
        result = narrow_param_space(specs, best, shrink_ratio=0.3)
        r = result[0]
        orig_range = 15.0 - 2.0  # 13.0
        # 15.0 - 14.29 = 0.71 < 13.0 * 0.2 = 2.6 → near high boundary
        assert abs(r["high"] - 15.0) < 1e-9
        assert abs(r["low"] - (15.0 - 13.0 * 0.4)) < 1e-9  # 15 - 5.2 = 9.8

    def test_missing_param_unchanged(self):
        """Parameters not in best_params are returned unchanged."""
        specs = [_make_spec("x", 0.0, 10.0), _make_spec("y", 1.0, 5.0)]
        best = {"x": 5.0}  # y not in best
        result = narrow_param_space(specs, best, shrink_ratio=0.3)
        assert result[1]["low"] == 1.0
        assert result[1]["high"] == 5.0

    def test_value_just_outside_boundary_zone(self):
        """Value at exactly 20% from boundary: should NOT trigger boundary protection."""
        specs = [_make_spec("x", 0.0, 10.0)]
        # val=2.0: (2.0 - 0.0) = 2.0, which is NOT < 10.0 * 0.2 = 2.0
        best = {"x": 2.0}
        result = narrow_param_space(specs, best, shrink_ratio=0.3)
        r = result[0]
        # Normal shrink: half_range=1.5, [0.5, 3.5]
        assert abs(r["low"] - 0.5) < 1e-9
        assert abs(r["high"] - 3.5) < 1e-9


# =====================================================================
# optimize_two_phase tests
# =====================================================================

class TestOptimizeTwoPhaseProbeValidation:
    """Tests for the probe validation that skips fine phase on dead params."""

    def test_skips_fine_when_coarse_best_is_dead(self):
        """When coarse best params produce -10, fine phase should be skipped."""
        call_count = {"total": 0, "fine": False}

        def mock_objective(params, scoring_mode="tanh"):
            call_count["total"] += 1
            if scoring_mode == "linear":
                call_count["fine"] = True
            # Always return -10 (dead zone)
            return -10.0

        specs = [_make_spec("x", 0.0, 10.0)]
        result = optimize_two_phase(
            mock_objective, specs,
            coarse_trials=5, fine_trials=5,
            seed=42, verbose=False, probe_trials=3,
        )

        # Fine phase should never have been called
        assert call_count["fine"] is False
        # Result should indicate early termination
        assert result["fine_best"] is None
        assert result["best_value"] <= -5.0

    def test_proceeds_when_coarse_best_is_valid(self):
        """When coarse best params produce a valid score, fine phase should run."""
        call_count = {"fine_calls": 0}

        def mock_objective(params, scoring_mode="tanh"):
            if scoring_mode == "linear":
                call_count["fine_calls"] += 1
            # Return a valid score
            return 3.5

        specs = [_make_spec("x", 0.0, 10.0)]
        result = optimize_two_phase(
            mock_objective, specs,
            coarse_trials=5, fine_trials=5,
            seed=42, verbose=False, probe_trials=3,
        )

        # Fine phase should have run
        assert call_count["fine_calls"] > 0
        assert result["fine_best"] is not None
        assert result["phase"] == "two_phase"
