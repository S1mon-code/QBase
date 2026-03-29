"""Batch Attribution Orchestrator — run attribution for every strategy in a portfolio.

Loads a weights JSON (from portfolio/builder.py output), runs signal + regime
attribution for each strategy, generates individual Markdown reports, then calls
coverage.py to produce the regime coverage matrix.

Usage:
    python -m attribution.batch \
        --weights strategies/strong_trend/portfolio/weights_ag.json \
        --start 2025-01-01 --end 2026-03-01

    # All-time strategies
    python -m attribution.batch \
        --weights strategies/all_time/ag/portfolio/weights_ag.json \
        --start 2022-01-01 --end 2026-03-01
"""
import sys
import json
import argparse
import importlib
import importlib.util
import warnings
from pathlib import Path
from dataclasses import dataclass, field

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

warnings.filterwarnings("ignore")


# =========================================================================
# Weights JSON Parsing — handles all known formats
# =========================================================================

@dataclass
class StrategyEntry:
    """Normalised strategy entry extracted from a weights JSON."""
    version: str
    weight: float
    freq: str
    params: dict = field(default_factory=dict)
    role: str = ""


def _detect_strategy_dir(weights_path: str) -> str:
    """Infer the strategy directory from the weights file location.

    Walks up from the weights file looking for known directory patterns:
      strategies/strong_trend/portfolio/weights_ag.json  -> strong_trend
      strategies/all_time/ag/portfolio/weights_ag.json   -> all_time/ag
      strategies/all_time/i/portfolio/weights_i.json     -> all_time/i
    """
    p = Path(weights_path).resolve()
    parts = p.parts
    try:
        idx = parts.index("strategies")
    except ValueError:
        return "strong_trend"

    # e.g. (..., 'strategies', 'strong_trend', 'portfolio', 'weights_ag.json')
    # or   (..., 'strategies', 'all_time', 'ag', 'portfolio', 'weights_ag.json')
    after = parts[idx + 1:]  # ('strong_trend', 'portfolio', ...)
    if len(after) >= 3 and after[0] == "all_time":
        return f"all_time/{after[1]}"
    return after[0] if after else "strong_trend"


def _load_optimization_results(strategy_dir: str) -> dict:
    """Load optimization_results.json for a strategy directory.

    Returns a dict mapping version -> best_params.
    """
    opt_path = Path(QBASE_ROOT) / "strategies" / strategy_dir / "optimization_results.json"
    if not opt_path.exists():
        return {}
    with open(opt_path) as f:
        data = json.load(f)

    result = {}
    if isinstance(data, list):
        for entry in data:
            ver = entry.get("version", "")
            params = entry.get("best_params", {})
            if ver and params:
                result[ver] = params
    elif isinstance(data, dict):
        # Hypothetical dict format
        for ver, info in data.items():
            if isinstance(info, dict) and "best_params" in info:
                result[ver] = info["best_params"]
    return result


def parse_weights_json(weights_path: str) -> tuple[list[StrategyEntry], dict]:
    """Parse a weights JSON file, returning (strategies, meta).

    Handles three known formats:
    1. Builder dict format: {"strategies": {"v12": {"weight": ..., "params": {...}}}}
    2. List format with params: {"strategies": [{"version": "v12", "weight": ..., "params": {...}}]}
    3. List format without params: {"strategies": [{"version": "v12", "weight": ..., "freq": ...}]}

    When params are missing from the JSON, falls back to optimization_results.json.
    """
    with open(weights_path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    raw = data.get("strategies", {})
    entries = []

    if isinstance(raw, dict):
        # Format 1: dict keyed by version
        for version, info in raw.items():
            entries.append(StrategyEntry(
                version=version,
                weight=info.get("weight", 0),
                freq=info.get("freq", "daily"),
                params=info.get("params", {}),
                role=info.get("role", ""),
            ))
    elif isinstance(raw, list):
        # Format 2/3: list of dicts
        for item in raw:
            entries.append(StrategyEntry(
                version=item.get("version", ""),
                weight=item.get("weight", 0),
                freq=item.get("freq", "daily"),
                params=item.get("params", {}),
                role=item.get("role", ""),
            ))

    # Fill missing params from optimization_results.json
    needs_params = [e for e in entries if not e.params]
    if needs_params:
        strategy_dir = _detect_strategy_dir(weights_path)
        opt_params = _load_optimization_results(strategy_dir)
        for entry in needs_params:
            if entry.version in opt_params:
                entry.params = opt_params[entry.version]

    return entries, meta


# =========================================================================
# Strategy Class Loading — supports strong_trend and all_time/*
# =========================================================================

# Strong trend class name mapping (mirrors strong_trend/optimizer.py)
_STRONG_TREND_CLASSES = {f"v{i}": f"StrongTrendV{i}" for i in range(1, 51)}
_STRONG_TREND_CLASSES["v3"] = "DonchianADXChandelierStrategy"


def load_strategy_class(version: str, strategy_dir: str):
    """Dynamically load a strategy class.

    Args:
        version: e.g. "v12"
        strategy_dir: e.g. "strong_trend" or "all_time/ag"
    """
    if strategy_dir == "strong_trend":
        mod = importlib.import_module(f"strategies.strong_trend.{version}")
        class_name = _STRONG_TREND_CLASSES.get(version, f"StrongTrendV{version[1:]}")
        return getattr(mod, class_name)
    else:
        # all_time: load via file and find TimeSeriesStrategy subclass
        filepath = Path(QBASE_ROOT) / "strategies" / strategy_dir / f"{version}.py"
        if not filepath.exists():
            raise FileNotFoundError(f"Strategy file not found: {filepath}")

        spec = importlib.util.spec_from_file_location(f"strategy_{strategy_dir}_{version}", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from alphaforge.strategy.base import TimeSeriesStrategy
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, TimeSeriesStrategy)
                    and attr is not TimeSeriesStrategy):
                return attr
        raise ValueError(f"No TimeSeriesStrategy subclass found in {filepath}")


# =========================================================================
# Batch Attribution Runner
# =========================================================================

def run_batch_attribution(
    weights_path: str,
    symbol: str,
    start: str,
    end: str,
    output_dir: str | None = None,
) -> dict:
    """Run signal + regime attribution for every strategy in a portfolio weights JSON.

    Args:
        weights_path: Path to weights JSON file.
        symbol: Trading symbol (e.g. "AG").
        start: Backtest start date.
        end: Backtest end date.
        output_dir: Directory for reports (default: research_log/attribution/).

    Returns:
        dict with keys: 'signal_results', 'regime_results', 'failures', 'reports'.
    """
    from attribution.signal import run_signal_attribution
    from attribution.regime import run_regime_attribution
    from attribution.report import generate_attribution_report

    if output_dir is None:
        output_dir = str(Path(QBASE_ROOT) / "research_log" / "attribution")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    entries, meta = parse_weights_json(weights_path)
    strategy_dir = _detect_strategy_dir(weights_path)

    # Override symbol from meta if present and not explicitly specified
    if not symbol and meta.get("symbol"):
        symbol = meta["symbol"]

    signal_results = {}
    regime_results = {}
    failures = []
    reports = []

    print(f"\n{'='*60}")
    print(f"Batch Attribution: {len(entries)} strategies from {Path(weights_path).name}")
    print(f"Symbol: {symbol} | Period: {start} ~ {end} | Dir: {strategy_dir}")
    print(f"{'='*60}\n")

    for entry in entries:
        ver = entry.version
        weight_pct = entry.weight * 100

        if not entry.params:
            print(f"  [{ver}] ({weight_pct:.1f}%) SKIP — no params available")
            failures.append({"version": ver, "reason": "no params"})
            continue

        try:
            print(f"  [{ver}] ({weight_pct:.1f}%) Loading strategy...", end=" ", flush=True)
            strategy_cls = load_strategy_class(ver, strategy_dir)

            print("signal...", end=" ", flush=True)
            sig_result = run_signal_attribution(
                strategy_cls, entry.params, symbol, start, end, freq=entry.freq,
            )
            signal_results[ver] = sig_result

            print("regime...", end=" ", flush=True)
            reg_result = run_regime_attribution(
                strategy_cls, entry.params, symbol, start, end, freq=entry.freq,
            )
            regime_results[ver] = reg_result

            # Generate individual report
            report_path = str(Path(output_dir) / f"{ver}_{symbol}.md")
            generate_attribution_report(sig_result, reg_result, report_path)
            reports.append(report_path)

            # Compact summary
            trades = reg_result.total_trades
            sharpe = reg_result.total_sharpe
            dominant = sig_result.dominant_indicator or "N/A"
            print(f"done (trades={trades}, sharpe={sharpe:.2f}, dominant={dominant})")

        except Exception as e:
            print(f"FAILED ({e})")
            failures.append({"version": ver, "reason": str(e)})

    # Generate coverage matrix
    if regime_results:
        try:
            from attribution.coverage import generate_coverage_matrix
            coverage_md, coverage_json = generate_coverage_matrix(
                regime_results=regime_results,
                weights={e.version: e.weight for e in entries},
                symbol=symbol,
                output_dir=output_dir,
            )
            reports.append(coverage_md)
            print(f"\nCoverage matrix: {coverage_md}")
        except Exception as e:
            print(f"\nCoverage matrix generation FAILED: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {len(signal_results)}/{len(entries)} succeeded, "
          f"{len(failures)} failed")
    if failures:
        for f in failures:
            print(f"  FAILED: {f['version']} — {f['reason']}")
    print(f"Reports written to: {output_dir}")
    print(f"{'='*60}\n")

    return {
        "signal_results": signal_results,
        "regime_results": regime_results,
        "failures": failures,
        "reports": reports,
    }


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Attribution — run signal + regime attribution for a portfolio"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to portfolio weights JSON")
    parser.add_argument("--symbol", default="",
                        help="Override symbol (default: auto-detect from weights JSON)")
    parser.add_argument("--start", default="2025-01-01",
                        help="Backtest start date (default: 2025-01-01)")
    parser.add_argument("--end", default="2026-03-01",
                        help="Backtest end date (default: 2026-03-01)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: research_log/attribution/)")
    args = parser.parse_args()

    # Resolve symbol from weights meta if not provided
    symbol = args.symbol
    if not symbol:
        with open(args.weights) as f:
            data = json.load(f)
        symbol = data.get("meta", {}).get("symbol", "AG")

    run_batch_attribution(
        weights_path=args.weights,
        symbol=symbol,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
