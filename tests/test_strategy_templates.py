"""Tests for QBase non-ATR strategy templates (mean reversion, vol target, time-based)."""
import ast
import sys
from pathlib import Path

import pytest

QBASE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(QBASE_ROOT))


# =====================================================================
# Helpers (shared with test_templates.py pattern)
# =====================================================================

def _read_source(relpath: str) -> str:
    """Read source file relative to QBase root."""
    return (QBASE_ROOT / relpath).read_text()


def _find_method(tree: ast.Module, method_name: str):
    """Find a method definition by name in an AST tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def _check_spec_manager_module_level(relpath: str):
    """Verify _SPEC_MANAGER is assigned at module level."""
    source = _read_source(relpath)
    tree = ast.parse(source)
    module_assigns = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.Assign)
    ]
    return any(
        any(
            isinstance(t, ast.Name) and t.id == "_SPEC_MANAGER"
            for t in node.targets
        )
        for node in module_assigns
    )


def _check_is_rollover_correct(relpath: str):
    """Verify context.is_rollover is used (not context.current_bar.is_rollover)."""
    source = _read_source(relpath)
    has_correct = "context.is_rollover" in source
    has_antipattern = "context.current_bar.is_rollover" in source
    return has_correct and not has_antipattern


# =====================================================================
# 1. Mean Reversion Template
# =====================================================================

class TestTemplateMeanReversion:
    """Tests for template_mean_reversion.py."""

    def test_import(self):
        from strategies.template_mean_reversion import MeanReversionStrategy
        assert MeanReversionStrategy is not None

    def test_instantiate(self):
        from strategies.template_mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        assert s.name == "mean_reversion_strategy"
        assert s.warmup == 60
        assert s.freq == "daily"

    def test_has_required_methods(self):
        from strategies.template_mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        assert hasattr(s, "on_init")
        assert hasattr(s, "on_init_arrays")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "_calc_lots")
        assert hasattr(s, "_reset")

    def test_has_three_exit_types(self):
        """Mean reversion must have profit_target, stop_loss, and time_stop exits."""
        from strategies.template_mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        # Profit target parameter
        assert hasattr(s, "profit_target_atr"), "Must have profit_target_atr parameter"
        assert s.profit_target_atr > 0
        # Stop loss parameter
        assert hasattr(s, "stop_loss_atr"), "Must have stop_loss_atr parameter"
        assert s.stop_loss_atr > 0
        # Time stop parameter
        assert hasattr(s, "max_holding_bars"), "Must have max_holding_bars parameter"
        assert s.max_holding_bars > 0

    def test_mr_uses_smaller_risk(self):
        """Mean reversion should risk 1% (not 2% like trend)."""
        source = _read_source("strategies/template_mean_reversion.py")
        assert "0.01" in source, "MR should use 1% risk (0.01)"

    def test_mr_has_both_directions(self):
        """Mean reversion should support both long and short."""
        source = _read_source("strategies/template_mean_reversion.py")
        assert "context.buy(" in source
        assert "context.sell(" in source
        assert "context.close_long()" in source
        assert "context.close_short()" in source

    def test_mr_no_trailing_stop(self):
        """Mean reversion must NOT use a trailing stop."""
        source = _read_source("strategies/template_mean_reversion.py")
        assert "highest_since_entry" not in source, \
            "MR must use fixed stop, not trailing stop"

    def test_mr_stop_range(self):
        """MR stops should be tight: 1.5-2.5× ATR."""
        from strategies.template_mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        assert 1.5 <= s.stop_loss_atr <= 2.5, \
            f"MR stop should be 1.5-2.5× ATR, got {s.stop_loss_atr}"

    def test_tunable_parameters(self):
        from strategies.template_mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        assert hasattr(s, "rsi_period")
        assert hasattr(s, "oversold_level")
        assert hasattr(s, "overbought_level")
        assert s.oversold_level < 50 < s.overbought_level


# =====================================================================
# 2. Volatility Target Template
# =====================================================================

class TestTemplateVolatilityTarget:
    """Tests for template_volatility_target.py."""

    def test_import(self):
        from strategies.template_volatility_target import VolatilityTargetStrategy
        assert VolatilityTargetStrategy is not None

    def test_instantiate(self):
        from strategies.template_volatility_target import VolatilityTargetStrategy
        s = VolatilityTargetStrategy()
        assert s.name == "volatility_target_strategy"
        assert s.warmup == 60
        assert s.freq == "daily"

    def test_has_required_methods(self):
        from strategies.template_volatility_target import VolatilityTargetStrategy
        s = VolatilityTargetStrategy()
        assert hasattr(s, "on_init")
        assert hasattr(s, "on_init_arrays")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "_calc_lots_vol_target")
        assert hasattr(s, "_reset")

    def test_has_vol_target_parameter(self):
        """Volatility target template must have vol_target parameter."""
        from strategies.template_volatility_target import VolatilityTargetStrategy
        s = VolatilityTargetStrategy()
        assert hasattr(s, "vol_target"), "Must have vol_target parameter"
        assert 0.05 <= s.vol_target <= 0.50, \
            f"vol_target should be reasonable (5-50%), got {s.vol_target}"

    def test_has_percentage_stop(self):
        """Vol target template should use percentage stop, not ATR stop."""
        from strategies.template_volatility_target import VolatilityTargetStrategy
        s = VolatilityTargetStrategy()
        assert hasattr(s, "pct_stop"), "Must have pct_stop parameter"
        assert 0.01 <= s.pct_stop <= 0.10

    def test_has_max_leverage_cap(self):
        """Must have leverage cap to prevent blow-up in low-vol."""
        from strategies.template_volatility_target import VolatilityTargetStrategy
        s = VolatilityTargetStrategy()
        assert hasattr(s, "max_leverage"), "Must have max_leverage cap"
        assert s.max_leverage > 0

    def test_no_trailing_stop(self):
        """Vol target must NOT use ATR trailing stop."""
        source = _read_source("strategies/template_volatility_target.py")
        assert "highest_since_entry" not in source
        assert "atr_trail_mult" not in source


# =====================================================================
# 3. Time-Based Exit Template
# =====================================================================

class TestTemplateTimeBased:
    """Tests for template_time_based.py."""

    def test_import(self):
        from strategies.template_time_based import TimeBasedStrategy
        assert TimeBasedStrategy is not None

    def test_instantiate(self):
        from strategies.template_time_based import TimeBasedStrategy
        s = TimeBasedStrategy()
        assert s.name == "time_based_strategy"
        assert s.warmup == 60
        assert s.freq == "daily"

    def test_has_required_methods(self):
        from strategies.template_time_based import TimeBasedStrategy
        s = TimeBasedStrategy()
        assert hasattr(s, "on_init")
        assert hasattr(s, "on_init_arrays")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "_calc_lots")
        assert hasattr(s, "_reset")

    def test_has_hold_bars_parameter(self):
        """Time-based template must have hold_bars parameter."""
        from strategies.template_time_based import TimeBasedStrategy
        s = TimeBasedStrategy()
        assert hasattr(s, "hold_bars"), "Must have hold_bars parameter"
        assert s.hold_bars > 0
        assert isinstance(s.hold_bars, int)

    def test_has_emergency_stop(self):
        """Must have emergency percentage stop as safety net."""
        from strategies.template_time_based import TimeBasedStrategy
        s = TimeBasedStrategy()
        assert hasattr(s, "emergency_stop_pct"), "Must have emergency_stop_pct"
        assert 0.01 <= s.emergency_stop_pct <= 0.10

    def test_no_trailing_stop(self):
        """Time-based must NOT use ATR trailing stop."""
        source = _read_source("strategies/template_time_based.py")
        assert "highest_since_entry" not in source
        assert "atr_trail_mult" not in source

    def test_time_exit_in_on_bar(self):
        """on_bar must check bars_held >= hold_bars."""
        source = _read_source("strategies/template_time_based.py")
        assert "bars_held" in source
        assert "hold_bars" in source


# =====================================================================
# 4. V4 Compliance (all three templates)
# =====================================================================

TEMPLATES = [
    "strategies/template_mean_reversion.py",
    "strategies/template_volatility_target.py",
    "strategies/template_time_based.py",
]


class TestV4ComplianceNewTemplates:
    """Verify V4 patterns across all new templates."""

    @pytest.mark.parametrize("relpath", TEMPLATES)
    def test_spec_manager_module_level(self, relpath):
        """_SPEC_MANAGER must be at module level, not inside methods."""
        assert _check_spec_manager_module_level(relpath), \
            f"_SPEC_MANAGER must be at module level in {relpath}"

    @pytest.mark.parametrize("relpath", TEMPLATES)
    def test_is_rollover_correct(self, relpath):
        """Must use context.is_rollover, not context.current_bar.is_rollover."""
        assert _check_is_rollover_correct(relpath), \
            f"Must use context.is_rollover (not current_bar) in {relpath}"

    @pytest.mark.parametrize("relpath", TEMPLATES)
    def test_spec_manager_not_in_calc_lots(self, relpath):
        """_calc_lots must NOT instantiate ContractSpecManager."""
        source = _read_source(relpath)
        tree = ast.parse(source)
        # Find any method with "calc_lots" in name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "calc_lots" in node.name:
                body_source = ast.dump(node)
                assert "ContractSpecManager" not in body_source, \
                    f"calc_lots in {relpath} must NOT instantiate ContractSpecManager"

    @pytest.mark.parametrize("relpath", TEMPLATES)
    def test_uses_precompute_pattern(self, relpath):
        """All templates must use on_init_arrays precompute pattern."""
        source = _read_source(relpath)
        assert "on_init_arrays" in source, \
            f"{relpath} must use on_init_arrays precompute pattern"
        assert "context.bar_index" in source, \
            f"{relpath} must use context.bar_index for O(1) lookup"


# =====================================================================
# 5. Structural Diversity Check
# =====================================================================

class TestStructuralDiversity:
    """Verify the new templates are actually different from trend templates."""

    def test_no_shared_atr_trailing_stop(self):
        """None of the new templates should use ATR trailing stop pattern."""
        for relpath in TEMPLATES:
            source = _read_source(relpath)
            # The trailing stop pattern: highest_since_entry - mult * atr
            assert "highest_since_entry" not in source or \
                   "trailing" not in source, \
                f"{relpath} should NOT use ATR trailing stop pattern"

    def test_different_exit_mechanisms(self):
        """Each template should have a distinct primary exit mechanism."""
        mr = _read_source("strategies/template_mean_reversion.py")
        vt = _read_source("strategies/template_volatility_target.py")
        tb = _read_source("strategies/template_time_based.py")

        # MR: profit target + time stop
        assert "target_price" in mr
        assert "max_holding_bars" in mr

        # VT: percentage stop
        assert "pct_stop" in vt

        # TB: hold_bars time exit
        assert "hold_bars" in tb
        assert "emergency_stop_pct" in tb
