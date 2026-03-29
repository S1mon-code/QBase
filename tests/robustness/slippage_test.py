"""Slippage Sensitivity Test — re-run strategies at increasing slippage levels.

Measures how strategy performance degrades as execution costs rise.
Importable as a library or usable via CLI.

Usage:
    # Portfolio mode (all strategies in a weights JSON):
    python -m tests.robustness.slippage_test \
        --weights strategies/strong_trend/portfolio/weights_ag.json \
        --start 2025-01-01 --end 2026-03-01 \
        --levels 1.0,2.0,3.0,5.0

    # Single strategy mode:
    python -m tests.robustness.slippage_test \
        --strategy strategies/strong_trend/v12.py \
        --symbol AG --freq daily \
        --start 2025-01-01 --end 2026-03-01
"""
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field

QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401


# =========================================================================
# Data types
# =========================================================================

@dataclass
class SlippageResult:
    """Result for a single slippage level."""
    slippage_ticks: float
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int


@dataclass
class SlippageReport:
    """Full slippage sensitivity report for one strategy."""
    strategy_name: str
    symbol: str
    freq: str
    results: list  # list of SlippageResult
    verdict: str = ""
    degradation_at_2x: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "freq": self.freq,
            "verdict": self.verdict,
            "degradation_at_2x": self.degradation_at_2x,
            "levels": [
                {
                    "slippage_ticks": r.slippage_ticks,
                    "sharpe": r.sharpe,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "n_trades": r.n_trades,
                }
                for r in self.results
            ],
        }


# =========================================================================
# Verdict logic
# =========================================================================

DEFAULT_LEVELS = [1.0, 2.0, 3.0, 5.0]


def compute_degradation(baseline_sharpe: float, test_sharpe: float) -> float:
    """Compute percentage degradation vs baseline.

    Returns a float in [-1, ...] where -0.22 means 22% drop.
    Returns 0.0 if baseline is zero or negative.
    """
    if baseline_sharpe <= 0:
        return 0.0
    return (test_sharpe - baseline_sharpe) / baseline_sharpe


def compute_verdict(degradation_at_2x: float) -> str:
    """Determine slippage sensitivity verdict.

    Args:
        degradation_at_2x: Degradation ratio at 2x slippage (e.g. -0.22 for 22% drop).

    Returns:
        "LOW", "MODERATE", or "HIGH"
    """
    drop = abs(degradation_at_2x)
    if drop < 0.20:
        return "LOW"
    elif drop <= 0.40:
        return "MODERATE"
    else:
        return "HIGH"


def verdict_description(verdict: str) -> str:
    """Human-readable explanation for a verdict."""
    descs = {
        "LOW": "robust — Sharpe drops < 20% at 2x slippage",
        "MODERATE": "acceptable — Sharpe drops 20-40% at 2x slippage",
        "HIGH": "alpha mainly from low slippage assumption, consider removing",
    }
    return descs.get(verdict, "unknown")


# =========================================================================
# Core: run slippage sweep for a single strategy
# =========================================================================

def run_slippage_sweep(
    strategy_cls,
    params: dict,
    symbol: str,
    start: str,
    end: str,
    freq: str = "daily",
    levels: list[float] | None = None,
    initial_capital: int = 1_000_000,
) -> SlippageReport:
    """Run backtests at multiple slippage levels and return a SlippageReport.

    Args:
        strategy_cls: Strategy class (must accept params via setattr).
        params: Strategy parameters dict.
        symbol: Trading symbol.
        start: Backtest start date string.
        end: Backtest end date string.
        freq: Frequency string (daily, 1h, etc).
        levels: List of slippage_ticks values to test.
        initial_capital: Initial capital for backtests.

    Returns:
        SlippageReport with results for each level and verdict.
    """
    from strategies.optimizer_core import create_strategy_with_params, run_single_backtest

    if levels is None:
        levels = list(DEFAULT_LEVELS)

    results = []
    for ticks in levels:
        strategy = create_strategy_with_params(strategy_cls, params)
        bt = run_single_backtest(
            strategy=strategy,
            symbol=symbol,
            start=start,
            end=end,
            freq=freq,
            initial_capital=initial_capital,
            slippage_ticks=ticks,
        )
        results.append(SlippageResult(
            slippage_ticks=ticks,
            sharpe=bt["sharpe"] if bt["sharpe"] != -999.0 else 0.0,
            total_return=bt["total_return"] or 0.0,
            max_drawdown=bt["max_drawdown"] or 0.0,
            n_trades=bt["n_trades"] or 0,
        ))

    # Compute degradation vs baseline (first level)
    baseline = results[0].sharpe
    deg_2x = 0.0
    for r in results:
        if r.slippage_ticks == 2.0:
            deg_2x = compute_degradation(baseline, r.sharpe)
            break
    # If 2.0 not in levels, use second level
    if 2.0 not in [r.slippage_ticks for r in results] and len(results) > 1:
        deg_2x = compute_degradation(baseline, results[1].sharpe)

    strategy_name = strategy_cls.__name__
    verdict = compute_verdict(deg_2x)

    return SlippageReport(
        strategy_name=strategy_name,
        symbol=symbol,
        freq=freq,
        results=results,
        verdict=verdict,
        degradation_at_2x=deg_2x,
    )


# =========================================================================
# Portfolio mode
# =========================================================================

def run_portfolio_slippage_test(
    weights_path: str,
    start: str,
    end: str,
    levels: list[float] | None = None,
) -> list[SlippageReport]:
    """Run slippage sweep for every strategy in a portfolio weights JSON.

    Returns list of SlippageReport (one per strategy).
    """
    from attribution.batch import parse_weights_json, load_strategy_class, _detect_strategy_dir

    entries, meta = parse_weights_json(weights_path)
    strategy_dir = _detect_strategy_dir(weights_path)
    symbol = meta.get("symbol", "AG")

    reports = []
    for entry in entries:
        if not entry.params:
            print(f"  [{entry.version}] SKIP — no params")
            continue

        try:
            strategy_cls = load_strategy_class(entry.version, strategy_dir)
            print(f"  [{entry.version}] Running slippage sweep...", end=" ", flush=True)
            report = run_slippage_sweep(
                strategy_cls=strategy_cls,
                params=entry.params,
                symbol=symbol,
                start=start,
                end=end,
                freq=entry.freq,
                levels=levels,
            )
            reports.append(report)
            print(f"done — {report.verdict}")
        except Exception as e:
            print(f"FAILED ({e})")

    return reports


# =========================================================================
# Display
# =========================================================================

def format_report(report: SlippageReport) -> str:
    """Format a SlippageReport as a readable table."""
    lines = [f"Strategy: {report.strategy_name} ({report.symbol}, {report.freq})"]

    # Table header
    lines.append(
        f"{'Slippage':<14} {'Sharpe':>7} {'Return':>9} {'MaxDD':>8} {'Trades':>7} {'Degrade':>8}"
    )
    lines.append("-" * 60)

    baseline_sharpe = report.results[0].sharpe if report.results else 0.0

    for r in report.results:
        mult = r.slippage_ticks / report.results[0].slippage_ticks if report.results else 1.0
        label = f"{r.slippage_ticks:.1f}"
        if r is report.results[0]:
            label += " (base)"
        else:
            label += f" ({mult:.0f}x)"

        deg = compute_degradation(baseline_sharpe, r.sharpe)
        deg_str = "  —" if r is report.results[0] else f"{deg:+.0%}"

        lines.append(
            f"{label:<14} {r.sharpe:>7.2f} {r.total_return:>+8.1%} "
            f"{r.max_drawdown:>7.1%} {r.n_trades:>7d} {deg_str:>8}"
        )

    deg_pct = abs(report.degradation_at_2x) * 100
    lines.append(
        f"\nVerdict: {report.verdict} sensitivity "
        f"(Sharpe drops {deg_pct:.0f}% at 2x) — {verdict_description(report.verdict)}"
    )
    return "\n".join(lines)


# =========================================================================
# Save results
# =========================================================================

def save_results(reports: list[SlippageReport], symbol: str) -> str:
    """Save reports to research_log/robustness/slippage_{symbol}.json."""
    output_dir = Path(QBASE_ROOT) / "research_log" / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"slippage_{symbol.lower()}.json"

    data = {
        "symbol": symbol,
        "reports": [r.to_dict() for r in reports],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


# =========================================================================
# Single strategy file loading
# =========================================================================

def load_strategy_from_file(filepath: str):
    """Load a strategy class from a .py file path.

    Finds the first TimeSeriesStrategy subclass in the module.
    """
    from alphaforge.strategy.base import TimeSeriesStrategy

    path = Path(filepath).resolve()
    spec = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, TimeSeriesStrategy)
                and attr is not TimeSeriesStrategy):
            return attr
    raise ValueError(f"No TimeSeriesStrategy subclass found in {filepath}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Slippage Sensitivity Test — measure strategy robustness to execution costs"
    )
    # Mode selection
    parser.add_argument("--weights", default=None,
                        help="Path to portfolio weights JSON (portfolio mode)")
    parser.add_argument("--strategy", default=None,
                        help="Path to strategy .py file (single strategy mode)")
    parser.add_argument("--symbol", default="AG",
                        help="Symbol for single strategy mode (default: AG)")
    parser.add_argument("--freq", default="daily",
                        help="Frequency for single strategy mode (default: daily)")

    # Common args
    parser.add_argument("--start", default="2025-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2026-03-01", help="Backtest end date")
    parser.add_argument("--levels", default="1.0,2.0,3.0,5.0",
                        help="Comma-separated slippage levels (default: 1.0,2.0,3.0,5.0)")

    args = parser.parse_args()
    levels = [float(x) for x in args.levels.split(",")]

    if args.weights:
        # Portfolio mode
        with open(args.weights) as f:
            data = json.load(f)
        symbol = data.get("meta", {}).get("symbol", "AG")

        print(f"\nSlippage Sensitivity Test — Portfolio Mode")
        print(f"Weights: {args.weights}")
        print(f"Symbol: {symbol} | Period: {args.start} ~ {args.end}")
        print(f"Levels: {levels}")
        print("=" * 60)

        reports = run_portfolio_slippage_test(
            weights_path=args.weights,
            start=args.start,
            end=args.end,
            levels=levels,
        )

        for report in reports:
            print(f"\n{format_report(report)}\n")

        if reports:
            path = save_results(reports, symbol)
            print(f"\nResults saved to: {path}")

    elif args.strategy:
        # Single strategy mode — load from file, use default params
        strategy_cls = load_strategy_from_file(args.strategy)
        # Instantiate with no params to get defaults
        instance = strategy_cls()
        params = {}

        # Try to load params from optimization_results.json
        strategy_path = Path(args.strategy).resolve()
        version = strategy_path.stem  # e.g. "v12"
        opt_file = strategy_path.parent / "optimization_results.json"
        if opt_file.exists():
            with open(opt_file) as f:
                opt_data = json.load(f)
            if isinstance(opt_data, list):
                for entry in opt_data:
                    if entry.get("version") == version:
                        params = entry.get("best_params", {})
                        break
            elif isinstance(opt_data, dict) and version in opt_data:
                params = opt_data[version].get("best_params", {})

        print(f"\nSlippage Sensitivity Test — Single Strategy")
        print(f"Strategy: {strategy_cls.__name__} ({args.symbol}, {args.freq})")
        print(f"Period: {args.start} ~ {args.end}")
        print(f"Levels: {levels}")
        if params:
            print(f"Params: {params}")
        print("=" * 60)

        report = run_slippage_sweep(
            strategy_cls=strategy_cls,
            params=params,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            freq=args.freq,
            levels=levels,
        )

        print(f"\n{format_report(report)}\n")
        path = save_results([report], args.symbol)
        print(f"Results saved to: {path}")

    else:
        parser.error("Provide either --weights (portfolio mode) or --strategy (single mode)")


if __name__ == "__main__":
    main()
