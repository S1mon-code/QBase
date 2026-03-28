"""Portfolio Stress Test — Monte Carlo simulation for tail risk analysis.

Runs the portfolio backtest once to extract daily returns, then resamples
with replacement to build distributions of Sharpe, drawdown, and CVaR.

Usage:
    python -m tests.robustness.stress_test \
        --weights strategies/strong_trend/portfolio/weights_ag.json \
        --start 2025-01-01 --end 2026-03-01 \
        --n-simulations 1000
"""
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np


# =========================================================================
# Data types
# =========================================================================

@dataclass
class StressTestResult:
    """Result of a Monte Carlo stress test."""
    symbol: str
    n_simulations: int
    base_sharpe: float
    mc_mean_sharpe: float
    mc_5th_pctl: float
    mc_95th_pctl: float
    prob_negative_sharpe: float  # 0-1
    cvar_95: float
    cvar_99: float
    max_sim_dd: float
    max_sim_dd_99: float  # 99th percentile of max drawdowns
    verdict: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_simulations": self.n_simulations,
            "base_sharpe": self.base_sharpe,
            "mc_mean_sharpe": self.mc_mean_sharpe,
            "mc_5th_pctl": self.mc_5th_pctl,
            "mc_95th_pctl": self.mc_95th_pctl,
            "prob_negative_sharpe": self.prob_negative_sharpe,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "max_sim_dd": self.max_sim_dd,
            "max_sim_dd_99": self.max_sim_dd_99,
            "verdict": self.verdict,
        }


# =========================================================================
# Monte Carlo core
# =========================================================================

def compute_sharpe_from_returns(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Compute annualized Sharpe ratio from daily returns array."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from daily returns array.

    Returns a negative float (e.g. -0.15 for 15% drawdown).
    """
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(np.min(drawdowns))


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Daily returns array.
        confidence: Confidence level (0.95 or 0.99).

    Returns:
        CVaR as a negative float (e.g. -0.021 for 2.1% expected loss).
    """
    if len(returns) == 0:
        return 0.0
    alpha = 1.0 - confidence
    sorted_returns = np.sort(returns)
    cutoff_idx = max(1, int(len(sorted_returns) * alpha))
    tail = sorted_returns[:cutoff_idx]
    return float(np.mean(tail))


def run_monte_carlo(
    daily_returns: np.ndarray,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> dict:
    """Run Monte Carlo resampling on daily returns.

    Args:
        daily_returns: Array of daily returns from actual backtest.
        n_simulations: Number of bootstrap simulations.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: sharpes, max_drawdowns (arrays of length n_simulations)
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_returns)

    sharpes = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Resample daily returns with replacement
        resampled = rng.choice(daily_returns, size=n_days, replace=True)
        sharpes[i] = compute_sharpe_from_returns(resampled)
        max_drawdowns[i] = compute_max_drawdown(resampled)

    return {
        "sharpes": sharpes,
        "max_drawdowns": max_drawdowns,
    }


# =========================================================================
# Verdict
# =========================================================================

def compute_stress_verdict(prob_neg_sharpe: float, cvar_95: float) -> str:
    """Determine stress test verdict.

    Args:
        prob_neg_sharpe: Probability of negative Sharpe (0-1).
        cvar_95: CVaR at 95% confidence (negative float).

    Returns:
        "ROBUST", "ACCEPTABLE", or "FRAGILE"
    """
    if prob_neg_sharpe < 0.05 and cvar_95 > -0.03:
        return "ROBUST"
    elif prob_neg_sharpe < 0.15 and cvar_95 > -0.05:
        return "ACCEPTABLE"
    else:
        return "FRAGILE"


def verdict_description(verdict: str) -> str:
    descs = {
        "ROBUST": "P(Sharpe<0) < 5% and CVaR 95% > -3%",
        "ACCEPTABLE": "P(Sharpe<0) < 15% and CVaR 95% > -5%",
        "FRAGILE": "high probability of negative Sharpe or extreme tail risk",
    }
    return descs.get(verdict, "unknown")


# =========================================================================
# Full stress test
# =========================================================================

def run_stress_test(
    daily_returns: np.ndarray,
    symbol: str = "UNKNOWN",
    n_simulations: int = 1000,
    seed: int | None = None,
) -> StressTestResult:
    """Run full stress test from daily returns.

    Args:
        daily_returns: Array of daily portfolio returns.
        symbol: Symbol name for reporting.
        n_simulations: Number of MC simulations.
        seed: Random seed.

    Returns:
        StressTestResult with all metrics and verdict.
    """
    base_sharpe = compute_sharpe_from_returns(daily_returns)

    mc = run_monte_carlo(daily_returns, n_simulations=n_simulations, seed=seed)
    sharpes = mc["sharpes"]
    max_dds = mc["max_drawdowns"]

    mc_mean = float(np.mean(sharpes))
    mc_5th = float(np.percentile(sharpes, 5))
    mc_95th = float(np.percentile(sharpes, 95))
    prob_neg = float(np.mean(sharpes < 0))

    cvar_95 = compute_cvar(daily_returns, 0.95)
    cvar_99 = compute_cvar(daily_returns, 0.99)

    max_sim_dd = float(np.min(max_dds))  # worst simulated drawdown
    max_sim_dd_99 = float(np.percentile(max_dds, 1))  # 99th percentile (1st pctl of neg values)

    verdict = compute_stress_verdict(prob_neg, cvar_95)

    return StressTestResult(
        symbol=symbol,
        n_simulations=n_simulations,
        base_sharpe=base_sharpe,
        mc_mean_sharpe=mc_mean,
        mc_5th_pctl=mc_5th,
        mc_95th_pctl=mc_95th,
        prob_negative_sharpe=prob_neg,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        max_sim_dd=max_sim_dd,
        max_sim_dd_99=max_sim_dd_99,
        verdict=verdict,
    )


# =========================================================================
# Portfolio daily returns extraction
# =========================================================================

def extract_portfolio_daily_returns(
    weights_path: str,
    start: str,
    end: str,
) -> tuple[np.ndarray, str]:
    """Run portfolio backtest and extract daily returns.

    Args:
        weights_path: Path to portfolio weights JSON.
        start: Backtest start date.
        end: Backtest end date.

    Returns:
        (daily_returns_array, symbol)
    """
    import pandas as pd
    from attribution.batch import parse_weights_json, load_strategy_class, _detect_strategy_dir
    from strategies.optimizer_core import (
        create_strategy_with_params, run_single_backtest,
        map_freq, resample_bars, _build_backtest_config,
    )
    from alphaforge.data.market import MarketDataLoader
    from alphaforge.data.contract_specs import ContractSpecManager
    from alphaforge.engine.event_driven import EventDrivenBacktester
    from config import get_data_dir

    entries, meta = parse_weights_json(weights_path)
    strategy_dir = _detect_strategy_dir(weights_path)
    symbol = meta.get("symbol", "AG")
    data_dir = get_data_dir()

    # Collect weighted daily returns from each strategy
    equity_curves = {}
    weights = {}

    for entry in entries:
        if not entry.params:
            continue
        try:
            strategy_cls = load_strategy_class(entry.version, strategy_dir)
            strategy = create_strategy_with_params(strategy_cls, entry.params)

            loader = MarketDataLoader(data_dir)
            load_freq, resample_factor = map_freq(entry.freq)
            bars = loader.load(symbol, freq=load_freq, start=start, end=end)
            if bars is None or len(bars) < strategy.warmup + 20:
                continue

            if resample_factor > 1:
                bars = resample_bars(bars, resample_factor)
            if bars is None or len(bars) < strategy.warmup + 20:
                continue

            # V6: Try to use BacktestConfig if available
            bt_config = _build_backtest_config("basic", 1_000_000, 1.0, None)
            if bt_config is not None:
                engine = EventDrivenBacktester(
                    spec_manager=ContractSpecManager(),
                    config=bt_config,
                )
            else:
                # Legacy fallback (AlphaForge < V6)
                engine = EventDrivenBacktester(
                    spec_manager=ContractSpecManager(),
                    initial_capital=1_000_000,
                    slippage_ticks=1.0,
                )
            result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)

            eq = getattr(result, "equity_curve", None)
            if eq is None:
                eq = getattr(result, "equity", None)
            if eq is not None and len(eq) > 0:
                if isinstance(eq, pd.Series):
                    daily_eq = eq.resample("D").last().dropna()
                else:
                    daily_eq = pd.Series(eq)
                equity_curves[entry.version] = daily_eq
                weights[entry.version] = entry.weight

        except Exception as e:
            print(f"  [{entry.version}] FAILED extracting equity: {e}")

    if not equity_curves:
        raise ValueError("No equity curves extracted from portfolio")

    # Normalize weights
    total_w = sum(weights.values())
    for k in weights:
        weights[k] /= total_w

    # Compute weighted portfolio returns
    all_returns = {}
    for ver, eq in equity_curves.items():
        rets = eq.pct_change().dropna()
        all_returns[ver] = rets

    # Align on common dates and compute weighted sum
    df = pd.DataFrame(all_returns)
    df = df.dropna(how="all").fillna(0.0)

    portfolio_returns = np.zeros(len(df))
    for ver in df.columns:
        portfolio_returns += df[ver].values * weights[ver]

    return portfolio_returns, symbol


# =========================================================================
# Display
# =========================================================================

def format_stress_result(result: StressTestResult) -> str:
    """Format a StressTestResult as a readable table."""
    lines = [
        f"Portfolio Stress Test: {result.symbol} ({result.n_simulations} simulations)",
        "",
        f"{'Metric':<20} {'Value':>10}",
        "-" * 32,
        f"{'Base Sharpe':<20} {result.base_sharpe:>10.2f}",
        f"{'MC Mean Sharpe':<20} {result.mc_mean_sharpe:>10.2f}",
        f"{'MC 5th pctl':<20} {result.mc_5th_pctl:>10.2f}",
        f"{'MC 95th pctl':<20} {result.mc_95th_pctl:>10.2f}",
        f"{'P(Sharpe < 0)':<20} {result.prob_negative_sharpe:>9.1%}",
        f"{'CVaR 95%':<20} {result.cvar_95:>9.1%}",
        f"{'CVaR 99%':<20} {result.cvar_99:>9.1%}",
        f"{'Max Sim DD':<20} {result.max_sim_dd:>9.1%}",
        f"{'Max Sim DD (99%)':<20} {result.max_sim_dd_99:>9.1%}",
        "",
        f"Verdict: {result.verdict} — {verdict_description(result.verdict)}",
    ]
    return "\n".join(lines)


# =========================================================================
# Save results
# =========================================================================

def save_results(result: StressTestResult) -> str:
    """Save stress test result to research_log/robustness/stress_test_{symbol}.json."""
    output_dir = Path(QBASE_ROOT) / "research_log" / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"stress_test_{result.symbol.lower()}.json"

    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return str(path)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Stress Test — Monte Carlo simulation for tail risk analysis"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to portfolio weights JSON")
    parser.add_argument("--start", default="2025-01-01",
                        help="Backtest start date (default: 2025-01-01)")
    parser.add_argument("--end", default="2026-03-01",
                        help="Backtest end date (default: 2026-03-01)")
    parser.add_argument("--n-simulations", type=int, default=1000,
                        help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"\nPortfolio Stress Test")
    print(f"Weights: {args.weights}")
    print(f"Period: {args.start} ~ {args.end}")
    print(f"Simulations: {args.n_simulations}")
    print("=" * 60)

    print("\nStep 1: Extracting portfolio daily returns...")
    daily_returns, symbol = extract_portfolio_daily_returns(
        weights_path=args.weights,
        start=args.start,
        end=args.end,
    )
    print(f"  Got {len(daily_returns)} daily returns for {symbol}")

    print("\nStep 2: Running Monte Carlo simulation...")
    result = run_stress_test(
        daily_returns=daily_returns,
        symbol=symbol,
        n_simulations=args.n_simulations,
        seed=args.seed,
    )

    print(f"\n{format_stress_result(result)}\n")

    path = save_results(result)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
