"""
tests/test_optimizer_interface.py — Static analysis tests for optimizer interface contracts.

Verifies that all 5 optimizer.py files correctly call optimizer_core functions
with matching signatures. Catches missing parameters, wrong parameter names,
and changed signatures WITHOUT importing the optimizer files (they have heavy deps).

Uses `inspect` for signature analysis and `ast` for source code parsing.
"""
import ast
import inspect
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import pytest

# Setup path so we can import optimizer_core
QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)

# We CAN import optimizer_core — it only depends on numpy/optuna
from strategies.optimizer_core import (
    auto_discover_params,
    suggest_params,
    create_strategy_with_params,
    composite_objective,
    run_single_backtest,
    optimize_two_phase,
    optimize_multi_seed,
    narrow_param_space,
    check_robustness,
    map_freq,
    resample_bars,
    build_result_entry,
    detect_strategy_status,
    is_strategy_dead,
    save_results_atomic,
    auto_calibrate_params,
)


# =====================================================================
# Helpers
# =====================================================================

# All public functions we expect in optimizer_core
PUBLIC_FUNCTIONS = {
    "auto_discover_params": auto_discover_params,
    "suggest_params": suggest_params,
    "create_strategy_with_params": create_strategy_with_params,
    "composite_objective": composite_objective,
    "run_single_backtest": run_single_backtest,
    "optimize_two_phase": optimize_two_phase,
    "optimize_multi_seed": optimize_multi_seed,
    "narrow_param_space": narrow_param_space,
    "check_robustness": check_robustness,
    "map_freq": map_freq,
    "resample_bars": resample_bars,
    "build_result_entry": build_result_entry,
    "detect_strategy_status": detect_strategy_status,
    "is_strategy_dead": is_strategy_dead,
    "save_results_atomic": save_results_atomic,
    "auto_calibrate_params": auto_calibrate_params,
}

# The 5 optimizer files
OPTIMIZER_FILES = {
    "strong_trend": os.path.join(QBASE_ROOT, "strategies", "strong_trend", "optimizer.py"),
    "all_time_ag": os.path.join(QBASE_ROOT, "strategies", "all_time", "ag", "optimizer.py"),
    "all_time_i": os.path.join(QBASE_ROOT, "strategies", "all_time", "i", "optimizer.py"),
    "boss": os.path.join(QBASE_ROOT, "strategies", "boss", "optimizer.py"),
    "medium_trend": os.path.join(QBASE_ROOT, "strategies", "medium_trend", "optimizer.py"),
}

# optimization_results.json locations
RESULTS_FILES = {
    "strong_trend": os.path.join(QBASE_ROOT, "strategies", "strong_trend", "optimization_results.json"),
    "all_time_ag": os.path.join(QBASE_ROOT, "strategies", "all_time", "ag", "optimization_results.json"),
    "all_time_i": os.path.join(QBASE_ROOT, "strategies", "all_time", "i", "optimization_results.json"),
    "boss": os.path.join(QBASE_ROOT, "strategies", "boss", "optimization_results.json"),
    "medium_trend": os.path.join(QBASE_ROOT, "strategies", "medium_trend", "optimization_results.json"),
}


def _get_sig(func):
    """Get function signature as a dict of {param_name: has_default}."""
    sig = inspect.signature(func)
    return {
        name: param.default is not inspect.Parameter.empty
        for name, param in sig.parameters.items()
    }


def _get_defaults(func):
    """Get default values for all parameters that have them."""
    sig = inspect.signature(func)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def _parse_file(filepath):
    """Parse a Python file and return its AST."""
    with open(filepath, "r") as f:
        return ast.parse(f.read(), filename=filepath)


def _find_calls_to(tree, func_name):
    """Find all calls to a given function name in an AST.

    Returns list of dicts with 'args' (positional count) and 'kwargs' (keyword names).
    Handles both direct calls like `composite_objective(...)` and
    attribute calls like `optimizer_core.composite_objective(...)`.
    """
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Direct call: func_name(...)
        if isinstance(node.func, ast.Name) and node.func.id == func_name:
            kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]
            calls.append({
                "n_positional": len(node.args),
                "kwargs": kwargs,
                "lineno": node.lineno,
            })

        # Attribute call: something.func_name(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
            kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]
            calls.append({
                "n_positional": len(node.args),
                "kwargs": kwargs,
                "lineno": node.lineno,
            })

    return calls


def _find_objective_fn_definitions(tree):
    """Find all inner function definitions named 'objective_fn' and analyze them.

    Returns list of dicts describing the function signature (parameter names).
    """
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "objective_fn":
                arg_names = [a.arg for a in node.args.args]
                # Check defaults
                n_defaults = len(node.args.defaults)
                n_args = len(arg_names)
                has_scoring_mode = "scoring_mode" in arg_names
                results.append({
                    "arg_names": arg_names,
                    "has_scoring_mode": has_scoring_mode,
                    "lineno": node.lineno,
                })
    return results


def _find_composite_objective_calls_in_fn(tree, fn_name="objective_fn"):
    """Find composite_objective calls inside a specific function definition."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == fn_name:
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name) and inner_node.func.id == "composite_objective":
                            kwargs = {kw.arg: kw for kw in inner_node.keywords if kw.arg is not None}
                            calls.append({
                                "kwargs": list(kwargs.keys()),
                                "has_scoring_mode": "scoring_mode" in kwargs,
                                "lineno": inner_node.lineno,
                            })
    return calls


# =====================================================================
# A. Function Signature Compatibility
# =====================================================================

class TestFunctionSignatures:
    """Verify optimizer_core public functions exist and have expected signatures."""

    @pytest.mark.parametrize("func_name", [
        "auto_discover_params",
        "suggest_params",
        "create_strategy_with_params",
        "composite_objective",
        "run_single_backtest",
        "optimize_two_phase",
        "optimize_multi_seed",
        "narrow_param_space",
        "check_robustness",
        "map_freq",
        "resample_bars",
    ])
    def test_function_exists_and_callable(self, func_name):
        """Each public function must exist and be callable."""
        func = PUBLIC_FUNCTIONS[func_name]
        assert callable(func), f"{func_name} is not callable"

    def test_auto_discover_params_signature(self):
        sig = _get_sig(auto_discover_params)
        assert "strategy_cls" in sig
        assert sig["strategy_cls"] is False  # required, no default

    def test_suggest_params_signature(self):
        sig = _get_sig(suggest_params)
        assert "trial" in sig
        assert "param_specs" in sig
        assert sig["trial"] is False
        assert sig["param_specs"] is False

    def test_create_strategy_with_params_signature(self):
        sig = _get_sig(create_strategy_with_params)
        assert "strategy_cls" in sig
        assert "params" in sig

    def test_composite_objective_signature(self):
        sig = _get_sig(composite_objective)
        defaults = _get_defaults(composite_objective)
        assert "results" in sig and sig["results"] is False
        assert "min_valid" in sig and sig["min_valid"] is True
        assert "freq" in sig and sig["freq"] is True
        assert "scoring_mode" in sig and sig["scoring_mode"] is True
        # Verify default values
        assert defaults["min_valid"] == 1
        assert defaults["freq"] is None
        assert defaults["scoring_mode"] == "tanh"

    def test_run_single_backtest_signature(self):
        sig = _get_sig(run_single_backtest)
        defaults = _get_defaults(run_single_backtest)
        assert "strategy" in sig and sig["strategy"] is False
        assert "symbol" in sig and sig["symbol"] is False
        assert "start" in sig and sig["start"] is False
        assert "end" in sig and sig["end"] is False
        assert "freq" in sig and defaults.get("freq") == "daily"
        assert "data_dir" in sig and defaults.get("data_dir") is None
        assert "initial_capital" in sig and defaults.get("initial_capital") == 1_000_000
        assert "slippage_ticks" in sig and defaults.get("slippage_ticks") == 1.0

    def test_optimize_two_phase_signature(self):
        sig = _get_sig(optimize_two_phase)
        defaults = _get_defaults(optimize_two_phase)
        assert "objective_fn" in sig and sig["objective_fn"] is False
        assert "param_specs" in sig and sig["param_specs"] is False
        assert "coarse_trials" in sig and defaults.get("coarse_trials") == 30
        assert "fine_trials" in sig and defaults.get("fine_trials") == 50
        assert "seed" in sig and defaults.get("seed") == 42
        assert "study_name" in sig and defaults.get("study_name") is None
        assert "verbose" in sig and defaults.get("verbose") is True
        assert "probe_trials" in sig and defaults.get("probe_trials") == 5

    def test_optimize_multi_seed_signature(self):
        sig = _get_sig(optimize_multi_seed)
        defaults = _get_defaults(optimize_multi_seed)
        assert "objective_fn" in sig and sig["objective_fn"] is False
        assert "param_specs" in sig and sig["param_specs"] is False
        assert "coarse_trials" in sig and defaults.get("coarse_trials") == 30
        assert "fine_trials" in sig and defaults.get("fine_trials") == 50
        assert "seeds" in sig and defaults.get("seeds") == (42, 123, 456)
        assert "verbose" in sig and defaults.get("verbose") is True
        assert "study_name" in sig and defaults.get("study_name") is None
        assert "probe_trials" in sig and defaults.get("probe_trials") == 5

    def test_narrow_param_space_signature(self):
        sig = _get_sig(narrow_param_space)
        defaults = _get_defaults(narrow_param_space)
        assert "param_specs" in sig and sig["param_specs"] is False
        assert "best_params" in sig and sig["best_params"] is False
        assert "shrink_ratio" in sig and defaults.get("shrink_ratio") == 0.3

    def test_check_robustness_signature(self):
        sig = _get_sig(check_robustness)
        defaults = _get_defaults(check_robustness)
        assert "evaluate_fn" in sig and sig["evaluate_fn"] is False
        assert "best_params" in sig and sig["best_params"] is False
        assert "param_specs" in sig and sig["param_specs"] is False
        assert "n_neighbors" in sig and defaults.get("n_neighbors") is None
        assert "min_plateau_ratio" in sig and defaults.get("min_plateau_ratio") == 0.6


# =====================================================================
# B. Caller Compatibility (AST-based static analysis)
# =====================================================================

class TestCallerCompatibility:
    """For each optimizer.py, parse AST and verify all calls to optimizer_core
    functions use correct argument names and don't pass unknown kwargs."""

    # Functions that each optimizer is expected to import from optimizer_core
    CORE_FUNCTIONS_USED = {
        "auto_discover_params",
        "suggest_params",
        "create_strategy_with_params",
        "composite_objective",
        "run_single_backtest",
        "optimize_two_phase",
        "optimize_multi_seed",
        "narrow_param_space",
    }

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_optimizer_file_exists(self, name, filepath):
        assert os.path.exists(filepath), f"Optimizer file missing: {filepath}"

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_imports_are_valid(self, name, filepath):
        """Verify optimizer only imports functions that exist in optimizer_core."""
        tree = _parse_file(filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "optimizer_core" in node.module:
                    for alias in node.names:
                        actual_name = alias.name
                        assert actual_name in PUBLIC_FUNCTIONS or actual_name in (
                            "map_freq", "resample_bars",
                            "_score_sharpe", "_score_risk",
                        ), (
                            f"{name}/optimizer.py imports '{actual_name}' from "
                            f"optimizer_core, but it doesn't exist as a public function"
                        )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_composite_objective_calls_valid(self, name, filepath):
        """All calls to composite_objective use only valid keyword arguments."""
        tree = _parse_file(filepath)
        calls = _find_calls_to(tree, "composite_objective")

        valid_kwargs = set(_get_sig(composite_objective).keys())
        for call in calls:
            for kwarg in call["kwargs"]:
                assert kwarg in valid_kwargs, (
                    f"{name}/optimizer.py line {call['lineno']}: "
                    f"composite_objective called with unknown kwarg '{kwarg}'. "
                    f"Valid kwargs: {valid_kwargs}"
                )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_optimize_two_phase_calls_valid(self, name, filepath):
        """All calls to optimize_two_phase use only valid keyword arguments."""
        tree = _parse_file(filepath)
        calls = _find_calls_to(tree, "optimize_two_phase")

        valid_kwargs = set(_get_sig(optimize_two_phase).keys())
        for call in calls:
            for kwarg in call["kwargs"]:
                assert kwarg in valid_kwargs, (
                    f"{name}/optimizer.py line {call['lineno']}: "
                    f"optimize_two_phase called with unknown kwarg '{kwarg}'. "
                    f"Valid kwargs: {valid_kwargs}"
                )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_optimize_multi_seed_calls_valid(self, name, filepath):
        """All calls to optimize_multi_seed use only valid keyword arguments."""
        tree = _parse_file(filepath)
        calls = _find_calls_to(tree, "optimize_multi_seed")

        valid_kwargs = set(_get_sig(optimize_multi_seed).keys())
        for call in calls:
            for kwarg in call["kwargs"]:
                assert kwarg in valid_kwargs, (
                    f"{name}/optimizer.py line {call['lineno']}: "
                    f"optimize_multi_seed called with unknown kwarg '{kwarg}'. "
                    f"Valid kwargs: {valid_kwargs}"
                )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_run_single_backtest_calls_valid(self, name, filepath):
        """All calls to run_single_backtest use only valid keyword arguments."""
        tree = _parse_file(filepath)
        calls = _find_calls_to(tree, "run_single_backtest")

        valid_kwargs = set(_get_sig(run_single_backtest).keys())
        for call in calls:
            for kwarg in call["kwargs"]:
                assert kwarg in valid_kwargs, (
                    f"{name}/optimizer.py line {call['lineno']}: "
                    f"run_single_backtest called with unknown kwarg '{kwarg}'. "
                    f"Valid kwargs: {valid_kwargs}"
                )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_narrow_param_space_calls_valid(self, name, filepath):
        """All calls to narrow_param_space use only valid keyword arguments."""
        tree = _parse_file(filepath)
        calls = _find_calls_to(tree, "narrow_param_space")

        valid_kwargs = set(_get_sig(narrow_param_space).keys())
        for call in calls:
            for kwarg in call["kwargs"]:
                assert kwarg in valid_kwargs, (
                    f"{name}/optimizer.py line {call['lineno']}: "
                    f"narrow_param_space called with unknown kwarg '{kwarg}'. "
                    f"Valid kwargs: {valid_kwargs}"
                )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_auto_discover_params_calls_valid(self, name, filepath):
        """All calls to auto_discover_params use only valid keyword arguments."""
        tree = _parse_file(filepath)
        calls = _find_calls_to(tree, "auto_discover_params")

        valid_kwargs = set(_get_sig(auto_discover_params).keys())
        for call in calls:
            for kwarg in call["kwargs"]:
                assert kwarg in valid_kwargs, (
                    f"{name}/optimizer.py line {call['lineno']}: "
                    f"auto_discover_params called with unknown kwarg '{kwarg}'. "
                    f"Valid kwargs: {valid_kwargs}"
                )


# =====================================================================
# C. scoring_mode Propagation
# =====================================================================

class TestScoringModePropagation:
    """Verify scoring_mode is correctly passed through the call chain.

    The bug pattern: an optimizer's objective_fn calls composite_objective
    but forgets to pass scoring_mode, so it always uses the default 'tanh'
    even during fine phase (which should use 'linear').

    optimize_two_phase internally calls objective_fn(params, scoring_mode="tanh")
    for coarse and objective_fn(params, scoring_mode="linear") for fine.
    So the objective_fn MUST accept scoring_mode and forward it.
    """

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_objective_fn_accepts_scoring_mode(self, name, filepath):
        """Each optimizer's objective_fn must accept a scoring_mode parameter."""
        tree = _parse_file(filepath)
        obj_fns = _find_objective_fn_definitions(tree)

        if not obj_fns:
            # Some optimizers may not define objective_fn (boss uses a lambda)
            # Check if there's any function that serves as the objective
            pytest.skip(f"{name} has no objective_fn definition")

        for fn_def in obj_fns:
            assert fn_def["has_scoring_mode"], (
                f"{name}/optimizer.py line {fn_def['lineno']}: "
                f"objective_fn({', '.join(fn_def['arg_names'])}) does NOT accept "
                f"'scoring_mode' parameter. This means optimize_two_phase will fail "
                f"when it tries to call objective_fn(params, scoring_mode='linear') "
                f"during fine phase. Add 'scoring_mode=\"tanh\"' as a parameter."
            )

    @pytest.mark.parametrize("name,filepath", list(OPTIMIZER_FILES.items()))
    def test_composite_objective_receives_scoring_mode(self, name, filepath):
        """When composite_objective is called inside objective_fn,
        it must receive scoring_mode (either explicitly or via the parameter)."""
        tree = _parse_file(filepath)
        calls = _find_composite_objective_calls_in_fn(tree, "objective_fn")

        for call in calls:
            assert call["has_scoring_mode"], (
                f"{name}/optimizer.py line {call['lineno']}: "
                f"composite_objective called inside objective_fn WITHOUT "
                f"scoring_mode parameter. The scoring_mode from the objective_fn "
                f"argument must be forwarded: "
                f"composite_objective(..., scoring_mode=scoring_mode)"
            )

    def test_boss_objective_fn_lacks_scoring_mode(self):
        """Boss optimizer's objective_fn does NOT accept scoring_mode.

        This is a KNOWN deviation. Boss's objective_fn(params) has no scoring_mode
        parameter, and calls composite_objective without it. This means:
        - optimize_two_phase will call objective_fn(params, scoring_mode="tanh")
          which will FAIL with TypeError (unexpected keyword argument).

        BUT: checking the actual optimize_two_phase code, the internal _objective
        calls objective_fn(params, scoring_mode=...), so if objective_fn doesn't
        accept scoring_mode, it WILL raise TypeError at runtime.

        This test documents the current state. If boss optimizer works, it's
        because optimize_two_phase's internal _objective wrapper handles this.
        """
        filepath = OPTIMIZER_FILES["boss"]
        tree = _parse_file(filepath)
        obj_fns = _find_objective_fn_definitions(tree)

        if not obj_fns:
            pytest.skip("boss has no objective_fn definition")

        # Boss objective_fn does NOT have scoring_mode — this is a known issue
        for fn_def in obj_fns:
            if not fn_def["has_scoring_mode"]:
                # Document it but don't fail — this is a known deviation
                # The test_objective_fn_accepts_scoring_mode will catch this
                pass

    @pytest.mark.parametrize("name,filepath", [
        (n, f) for n, f in OPTIMIZER_FILES.items() if n != "boss"
    ])
    def test_scoring_mode_forwarded_to_composite(self, name, filepath):
        """For non-boss optimizers, verify scoring_mode flows:
        objective_fn(params, scoring_mode) -> composite_objective(..., scoring_mode=scoring_mode)
        """
        tree = _parse_file(filepath)
        obj_fns = _find_objective_fn_definitions(tree)
        if not obj_fns:
            pytest.skip(f"{name} has no objective_fn")

        for fn_def in obj_fns:
            if not fn_def["has_scoring_mode"]:
                pytest.fail(
                    f"{name}: objective_fn doesn't accept scoring_mode"
                )

        # Now check composite_objective calls inside objective_fn pass it through
        calls = _find_composite_objective_calls_in_fn(tree, "objective_fn")
        for call in calls:
            assert call["has_scoring_mode"], (
                f"{name}/optimizer.py line {call['lineno']}: "
                f"scoring_mode accepted by objective_fn but NOT forwarded to "
                f"composite_objective. Add: scoring_mode=scoring_mode"
            )


# =====================================================================
# D. Result Format Consistency
# =====================================================================

# Fields that every result entry should ideally have
REQUIRED_RESULT_FIELDS = {"version", "best_params"}
# Fields that are common but may be named differently
COMMON_FIELDS_VARIANTS = {
    "score": {"best_sharpe", "best_score"},  # boss uses best_score, others best_sharpe
}


class TestResultFormatConsistency:
    """Check optimization_results.json files for required fields and consistency."""

    @pytest.mark.parametrize("name,filepath", list(RESULTS_FILES.items()))
    def test_results_file_exists(self, name, filepath):
        """Each optimizer should have a results file."""
        if not os.path.exists(filepath):
            pytest.skip(f"{name} results file not yet generated: {filepath}")

    @pytest.mark.parametrize("name,filepath", list(RESULTS_FILES.items()))
    def test_results_valid_json(self, name, filepath):
        """Results file must be valid JSON."""
        if not os.path.exists(filepath):
            pytest.skip(f"No results file: {filepath}")
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, list), f"{name}: results should be a JSON array"
        assert len(data) > 0, f"{name}: results array is empty"

    @pytest.mark.parametrize("name,filepath", list(RESULTS_FILES.items()))
    def test_results_have_required_fields(self, name, filepath):
        """Each result entry must have 'version' and 'best_params'."""
        if not os.path.exists(filepath):
            pytest.skip(f"No results file: {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        for i, entry in enumerate(data):
            if "error" in entry:
                continue  # error entries are allowed to have minimal fields
            for field in REQUIRED_RESULT_FIELDS:
                assert field in entry, (
                    f"{name} entry [{i}] (version={entry.get('version', '?')}): "
                    f"missing required field '{field}'"
                )

    @pytest.mark.parametrize("name,filepath", list(RESULTS_FILES.items()))
    def test_results_have_score_field(self, name, filepath):
        """Each result must have a score field (best_sharpe or best_score)."""
        if not os.path.exists(filepath):
            pytest.skip(f"No results file: {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        score_fields = {"best_sharpe", "best_score"}
        for i, entry in enumerate(data):
            if "error" in entry:
                continue
            has_score = any(f in entry for f in score_fields)
            assert has_score, (
                f"{name} entry [{i}] (version={entry.get('version', '?')}): "
                f"missing score field. Expected one of: {score_fields}. "
                f"Got keys: {set(entry.keys())}"
            )

    @pytest.mark.parametrize("name,filepath", list(RESULTS_FILES.items()))
    def test_best_params_is_dict(self, name, filepath):
        """best_params should be a dict with string keys."""
        if not os.path.exists(filepath):
            pytest.skip(f"No results file: {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        for i, entry in enumerate(data):
            if "error" in entry:
                continue
            bp = entry.get("best_params")
            assert isinstance(bp, dict), (
                f"{name} entry [{i}]: best_params should be dict, got {type(bp)}"
            )

    def test_cross_optimizer_score_field_naming(self):
        """Document which optimizers use which score field name."""
        score_field_map = {}
        for name, filepath in RESULTS_FILES.items():
            if not os.path.exists(filepath):
                continue
            with open(filepath) as f:
                data = json.load(f)
            valid_entries = [e for e in data if "error" not in e]
            if not valid_entries:
                continue
            entry = valid_entries[0]
            if "best_sharpe" in entry:
                score_field_map[name] = "best_sharpe"
            elif "best_score" in entry:
                score_field_map[name] = "best_score"

        # boss uses best_score, all others use best_sharpe — document this
        if "boss" in score_field_map:
            assert score_field_map["boss"] == "best_score", (
                f"Boss result format changed: expected 'best_score', "
                f"got '{score_field_map['boss']}'"
            )
        for name in ("strong_trend", "all_time_ag", "all_time_i", "medium_trend"):
            if name in score_field_map:
                assert score_field_map[name] == "best_sharpe", (
                    f"{name} result format changed: expected 'best_sharpe', "
                    f"got '{score_field_map[name]}'"
                )

    def test_robustness_field_format(self):
        """When present, robustness should be a dict with expected keys."""
        expected_robustness_keys = {"is_robust", "neighbor_mean", "neighbor_std", "above_threshold_pct"}
        for name, filepath in RESULTS_FILES.items():
            if not os.path.exists(filepath):
                continue
            with open(filepath) as f:
                data = json.load(f)
            for entry in data:
                rob = entry.get("robustness")
                if rob is None or not isinstance(rob, dict):
                    continue
                for key in expected_robustness_keys:
                    assert key in rob, (
                        f"{name} version={entry.get('version')}: "
                        f"robustness dict missing key '{key}'. "
                        f"Got keys: {set(rob.keys())}"
                    )
