"""Tests for QBase strategy templates."""
import ast
import sys
from pathlib import Path

import pytest

QBASE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(QBASE_ROOT))


# =====================================================================
# 1. Import and Instantiation Tests
# =====================================================================

class TestTemplateSimple:
    """Tests for template_simple.py."""

    def test_import(self):
        from strategies.template_simple import SimpleStrategy
        assert SimpleStrategy is not None

    def test_instantiate(self):
        from strategies.template_simple import SimpleStrategy
        s = SimpleStrategy()
        assert s.name == "simple_strategy"
        assert s.warmup == 60
        assert s.freq == "daily"

    def test_has_required_methods(self):
        from strategies.template_simple import SimpleStrategy
        s = SimpleStrategy()
        assert hasattr(s, "on_init")
        assert hasattr(s, "on_init_arrays")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "_calc_lots")
        assert hasattr(s, "_reset")

    def test_tunable_parameters(self):
        from strategies.template_simple import SimpleStrategy
        s = SimpleStrategy()
        assert hasattr(s, "indicator_period")
        assert hasattr(s, "atr_trail_mult")
        assert s.atr_trail_mult >= 4.0, "Simple template should use wide stop (>= 4.0 ATR)"


class TestTemplateFull:
    """Tests for template_full.py."""

    def test_import(self):
        from strategies.template_full import FullStrategy
        assert FullStrategy is not None

    def test_instantiate(self):
        from strategies.template_full import FullStrategy
        s = FullStrategy()
        assert s.name == "full_strategy"
        assert s.warmup == 60

    def test_has_required_methods(self):
        from strategies.template_full import FullStrategy
        s = FullStrategy()
        assert hasattr(s, "on_init")
        assert hasattr(s, "on_init_arrays")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "_calc_lots")
        assert hasattr(s, "_reset_state")
        assert hasattr(s, "_should_add")
        assert hasattr(s, "_calc_add_lots")

    def test_scale_factors(self):
        from strategies.template_full import SCALE_FACTORS, MAX_SCALE
        assert len(SCALE_FACTORS) == 3
        assert MAX_SCALE == 3
        assert SCALE_FACTORS[0] == 1.0


class TestTemplateBackwardCompat:
    """Tests for template.py backward compatibility re-export."""

    def test_import_template_strategy(self):
        from strategies.template import TemplateStrategy
        assert TemplateStrategy is not None

    def test_template_strategy_is_simple_strategy(self):
        from strategies.template import TemplateStrategy
        from strategies.template_simple import SimpleStrategy
        assert TemplateStrategy is SimpleStrategy

    def test_all_export(self):
        import strategies.template as mod
        assert hasattr(mod, "__all__")
        assert "TemplateStrategy" in mod.__all__


# =====================================================================
# 2. V4 Anti-Pattern Detection
# =====================================================================

class TestV4Correctness:
    """Verify V4 anti-patterns are fixed in templates."""

    def test_spec_manager_module_level_simple(self):
        """_SPEC_MANAGER must be at module level, not inside _calc_lots."""
        source = _read_source("strategies/template_simple.py")
        tree = ast.parse(source)

        # Check module-level assignment exists
        module_assigns = [
            node for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.Assign)
        ]
        spec_at_module = any(
            any(
                isinstance(t, ast.Name) and t.id == "_SPEC_MANAGER"
                for t in node.targets
            )
            for node in module_assigns
        )
        assert spec_at_module, "_SPEC_MANAGER must be assigned at module level"

        # Check _calc_lots does NOT create ContractSpecManager
        calc_lots = _find_method(tree, "_calc_lots")
        assert calc_lots is not None, "_calc_lots method must exist"
        body_source = ast.dump(calc_lots)
        assert "ContractSpecManager" not in body_source, \
            "_calc_lots must NOT instantiate ContractSpecManager"

    def test_spec_manager_module_level_full(self):
        """_SPEC_MANAGER must be at module level in template_full.py."""
        source = _read_source("strategies/template_full.py")
        tree = ast.parse(source)

        module_assigns = [
            node for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.Assign)
        ]
        spec_at_module = any(
            any(
                isinstance(t, ast.Name) and t.id == "_SPEC_MANAGER"
                for t in node.targets
            )
            for node in module_assigns
        )
        assert spec_at_module, "_SPEC_MANAGER must be assigned at module level"

    def test_is_rollover_direct_simple(self):
        """Must use context.is_rollover, not context.current_bar.is_rollover."""
        source = _read_source("strategies/template_simple.py")
        assert "context.is_rollover" in source, \
            "Must use context.is_rollover"
        assert "context.current_bar.is_rollover" not in source, \
            "Must NOT use context.current_bar.is_rollover"

    def test_is_rollover_direct_full(self):
        """Must use context.is_rollover in template_full.py."""
        source = _read_source("strategies/template_full.py")
        assert "context.is_rollover" in source
        assert "context.current_bar.is_rollover" not in source


# =====================================================================
# Helpers
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
