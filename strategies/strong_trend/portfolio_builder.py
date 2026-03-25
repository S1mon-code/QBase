"""
Portfolio Builder for Strong Trend Strategies
==============================================
1. Run all 50 strategies on AG test period → collect equity curves
2. Filter by test Sharpe > 0.3
3. Compute daily return correlation matrix
4. Correlation filter: > 0.7 → keep higher Sharpe
5. HRP weight allocation
6. Run PortfolioBacktester
7. Output results

Usage:
    python strategies/strong_trend/portfolio_builder.py
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.portfolio import (
    PortfolioConfig, StrategyAllocation, PortfolioBacktester,
)
from alphaforge.report import HTMLReportGenerator

from strategies.strong_trend.optimizer import (
    create_strategy_with_params, load_strategy_class, _map_freq, _resample_bars,
    DATA_DIR, STRATEGY_CLASSES,
)

# =========================================================================
# Configuration
# =========================================================================

TEST_SYMBOL = "AG"
TEST_START = "2025-01-01"
TEST_END = "2026-03-01"

# Also test on EC for validation
EC_SYMBOL = "EC"
EC_START = "2023-07-01"
EC_END = "2024-09-01"

SHARPE_THRESHOLD = 0.5      # Min mean test Sharpe to enter portfolio
CORR_THRESHOLD = 0.7        # Max pairwise correlation
TOTAL_CAPITAL = 3_000_000

RESULTS_PATH = str(Path(QBASE_ROOT) / "strategies" / "strong_trend" / "optimization_results.json")
OUTPUT_DIR = str(Path(QBASE_ROOT) / "strategies" / "strong_trend" / "portfolio")


# =========================================================================
# Step 1: Run individual backtests, collect equity curves
# =========================================================================

def run_single_test(version, params, symbol, start, end):
    """Run one strategy on one symbol, return (sharpe, daily_equity_series)."""
    try:
        strategy = create_strategy_with_params(version, params)
        freq = strategy.freq
        loader = MarketDataLoader(DATA_DIR)
        spec_manager = ContractSpecManager()

        load_freq, resample_factor = _map_freq(freq)
        bars = loader.load(symbol, freq=load_freq, start=start, end=end)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None, None

        if resample_factor > 1:
            bars = _resample_bars(bars, resample_factor)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None, None

        engine = EventDrivenBacktester(
            spec_manager=spec_manager,
            initial_capital=1_000_000,
            slippage_ticks=1.0,
        )
        result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)

        sharpe = result.sharpe
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None, None

        # Get equity curve as daily series
        eq = result.equity_curve
        if eq is not None and len(eq) > 0:
            if isinstance(eq, pd.Series):
                daily_eq = eq.resample("D").last().dropna()
            else:
                daily_eq = pd.Series(eq)
            return float(sharpe), daily_eq

        return float(sharpe), None
    except Exception as e:
        print(f"  {version} error: {e}")
        return None, None


def collect_all_results():
    """Run all 50 strategies on both test sets."""
    with open(RESULTS_PATH) as f:
        opt_results = json.load(f)

    print("=" * 70)
    print("STEP 1: Running individual backtests on AG + EC")
    print("=" * 70)

    all_data = []
    for r in opt_results:
        ver = r["version"]
        params = r["best_params"]
        train_sharpe = r["best_sharpe"]

        # AG test
        ag_sharpe, ag_equity = run_single_test(ver, params, TEST_SYMBOL, TEST_START, TEST_END)
        # EC test
        ec_sharpe, ec_equity = run_single_test(ver, params, EC_SYMBOL, EC_START, EC_END)

        mean_test = np.mean([s for s in [ag_sharpe, ec_sharpe] if s is not None]) if any(
            s is not None for s in [ag_sharpe, ec_sharpe]) else -999

        print(f"  {ver:>4s}: Train={train_sharpe:>7.3f}  AG={ag_sharpe or 0:>7.3f}  EC={ec_sharpe or 0:>7.3f}  Mean={mean_test:>7.3f}")

        all_data.append({
            "version": ver,
            "params": params,
            "train_sharpe": train_sharpe,
            "ag_sharpe": ag_sharpe,
            "ec_sharpe": ec_sharpe,
            "mean_test": mean_test,
            "ag_equity": ag_equity,
            "ec_equity": ec_equity,
            "freq": create_strategy_with_params(ver, params).freq,
        })

    return all_data


# =========================================================================
# Step 2: Filter by Sharpe
# =========================================================================

def filter_by_sharpe(all_data, threshold=SHARPE_THRESHOLD):
    """Keep strategies with mean test Sharpe > threshold."""
    filtered = [d for d in all_data if d["mean_test"] > threshold]
    print(f"\nSTEP 2: Sharpe filter (>{threshold}): {len(filtered)}/{len(all_data)} pass")
    for d in sorted(filtered, key=lambda x: x["mean_test"], reverse=True):
        print(f"  {d['version']:>4s}: Mean Test = {d['mean_test']:.3f}")
    return filtered


# =========================================================================
# Step 3: Correlation matrix + filter
# =========================================================================

def compute_correlation_matrix(filtered_data, symbol="ag"):
    """Compute daily return correlation matrix from equity curves."""
    equity_key = f"{symbol}_equity"
    returns_dict = {}

    for d in filtered_data:
        eq = d[equity_key]
        if eq is not None and len(eq) > 10:
            if isinstance(eq, pd.Series):
                ret = eq.pct_change().dropna()
            else:
                eq_s = pd.Series(eq)
                ret = eq_s.pct_change().dropna()
            if len(ret) > 5:
                returns_dict[d["version"]] = ret

    if len(returns_dict) < 2:
        print("  Not enough equity curves for correlation")
        return None, returns_dict

    # Align all returns to common index
    df = pd.DataFrame(returns_dict)
    df = df.dropna(how="all")
    # Forward fill short gaps
    df = df.fillna(0)

    corr = df.corr()
    return corr, returns_dict


def filter_by_correlation(filtered_data, corr_matrix, threshold=CORR_THRESHOLD):
    """Remove highly correlated strategies, keep higher Sharpe."""
    if corr_matrix is None:
        return filtered_data

    # Sort by mean_test descending
    sorted_data = sorted(filtered_data, key=lambda x: x["mean_test"], reverse=True)
    kept = []
    removed = set()

    for d in sorted_data:
        ver = d["version"]
        if ver in removed:
            continue
        if ver not in corr_matrix.columns:
            kept.append(d)
            continue

        kept.append(d)

        # Mark all lower-Sharpe strategies correlated > threshold with this one
        for d2 in sorted_data:
            v2 = d2["version"]
            if v2 == ver or v2 in removed or v2 not in corr_matrix.columns:
                continue
            if ver in corr_matrix.columns and v2 in corr_matrix.columns:
                c = abs(corr_matrix.loc[ver, v2])
                if c > threshold:
                    removed.add(v2)
                    print(f"  Remove {v2} (corr={c:.3f} with {ver})")

    print(f"\nSTEP 3: Correlation filter (>{threshold}): {len(kept)}/{len(filtered_data)} remain")
    for d in kept:
        print(f"  {d['version']:>4s}: Mean Test = {d['mean_test']:.3f}")
    return kept


# =========================================================================
# Step 4: HRP Weight Allocation
# =========================================================================

def hrp_weights(returns_dict, strategy_versions):
    """Hierarchical Risk Parity weight allocation."""
    # Build aligned returns DataFrame
    cols = [v for v in strategy_versions if v in returns_dict]
    if len(cols) < 2:
        # Equal weight fallback
        w = {v: 1.0 / len(strategy_versions) for v in strategy_versions}
        return w

    df = pd.DataFrame({v: returns_dict[v] for v in cols})
    df = df.fillna(0)

    cov = df.cov()
    corr = df.corr()

    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr))
    dist_condensed = squareform(dist.values, checks=False)

    # Hierarchical clustering
    link = linkage(dist_condensed, method="single")
    sort_ix = leaves_list(link)
    sorted_cols = [cols[i] for i in sort_ix]

    # Recursive bisection
    def _get_cluster_var(cov_mat, cluster_items):
        cov_sub = cov_mat.loc[cluster_items, cluster_items]
        w_ivp = 1.0 / np.diag(cov_sub)
        w_ivp = w_ivp / w_ivp.sum()
        return float(np.dot(w_ivp, np.dot(cov_sub.values, w_ivp)))

    def _recursive_bisect(cov_mat, sorted_items):
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = _get_cluster_var(cov_mat, left)
                var_right = _get_cluster_var(cov_mat, right)

                alpha = 1.0 - var_left / (var_left + var_right)

                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            clusters = new_clusters

        return weights / weights.sum()

    w = _recursive_bisect(cov, sorted_cols)

    # Add zero weight for strategies not in returns_dict
    result = {}
    for v in strategy_versions:
        result[v] = float(w.get(v, 0))

    # Renormalize
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result


# =========================================================================
# Step 5: Run Portfolio Backtest via AlphaForge
# =========================================================================

def run_portfolio_backtest(kept_data, weights, symbol, start, end, capital):
    """Run combined portfolio backtest."""
    print(f"\nSTEP 5: Portfolio backtest on {symbol} ({start} → {end})")

    allocations = []
    for d in kept_data:
        ver = d["version"]
        w = weights.get(ver, 0)
        if w <= 0.001:
            continue
        allocations.append(
            StrategyAllocation(
                name=f"strong_trend_{ver}",
                strategy_file=f"strategies/strong_trend/{ver}.py",
                symbols=symbol,
                freq=d["freq"],
                weight=w,
                params=d["params"],
            )
        )

    config = PortfolioConfig(
        total_capital=capital,
        allocations=allocations,
        rebalance="none",
    )

    runner = PortfolioBacktester(
        spec_manager=ContractSpecManager(),
        data_dir=DATA_DIR,
    )
    report = runner.run(config, start=start, end=end)
    return report


# =========================================================================
# Step 6: Risk Parity (alternative weighting)
# =========================================================================

def risk_parity_weights(returns_dict, strategy_versions):
    """Simple risk parity: weight inversely proportional to volatility."""
    vols = {}
    for v in strategy_versions:
        if v in returns_dict:
            ret = returns_dict[v]
            vol = ret.std()
            if vol > 0:
                vols[v] = vol

    if not vols:
        return {v: 1.0 / len(strategy_versions) for v in strategy_versions}

    inv_vols = {v: 1.0 / vol for v, vol in vols.items()}
    total = sum(inv_vols.values())
    return {v: inv_vols.get(v, 0) / total for v in strategy_versions}


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    # Step 1: Collect all results
    all_data = collect_all_results()

    # Step 2: Filter by Sharpe
    filtered = filter_by_sharpe(all_data, threshold=SHARPE_THRESHOLD)

    # Step 3: Correlation matrix + filter
    print(f"\nSTEP 3: Computing correlation matrix (AG equity curves)")
    corr_ag, returns_ag = compute_correlation_matrix(filtered, "ag")
    kept = filter_by_correlation(filtered, corr_ag, threshold=CORR_THRESHOLD)

    # Step 4: HRP weights
    print(f"\nSTEP 4: HRP weight allocation")
    versions_kept = [d["version"] for d in kept]
    hrp_w = hrp_weights(returns_ag, versions_kept)

    print("\n  HRP Weights:")
    for v in sorted(hrp_w, key=hrp_w.get, reverse=True):
        if hrp_w[v] > 0.001:
            sharpe = next((d["mean_test"] for d in kept if d["version"] == v), 0)
            print(f"    {v:>4s}: {hrp_w[v]*100:>5.1f}%  (Mean Test Sharpe = {sharpe:.3f})")

    # Step 4b: Risk Parity weights (alternative)
    rp_w = risk_parity_weights(returns_ag, versions_kept)
    print("\n  Risk Parity Weights:")
    for v in sorted(rp_w, key=rp_w.get, reverse=True):
        if rp_w[v] > 0.001:
            print(f"    {v:>4s}: {rp_w[v]*100:>5.1f}%")

    # Step 5a: HRP Portfolio on AG
    try:
        hrp_report_ag = run_portfolio_backtest(kept, hrp_w, TEST_SYMBOL, TEST_START, TEST_END, TOTAL_CAPITAL)
        print(f"\n  HRP Portfolio AG:")
        print(f"    Sharpe: {hrp_report_ag.combined_result.sharpe:.3f}")
        print(f"    Return: {hrp_report_ag.combined_result.total_return:.2%}")
        print(f"    MaxDD:  {hrp_report_ag.combined_result.max_drawdown:.2%}")
    except Exception as e:
        print(f"  HRP AG failed: {e}")
        hrp_report_ag = None

    # Step 5b: HRP Portfolio on EC
    try:
        hrp_report_ec = run_portfolio_backtest(kept, hrp_w, EC_SYMBOL, EC_START, EC_END, TOTAL_CAPITAL)
        print(f"\n  HRP Portfolio EC:")
        print(f"    Sharpe: {hrp_report_ec.combined_result.sharpe:.3f}")
        print(f"    Return: {hrp_report_ec.combined_result.total_return:.2%}")
        print(f"    MaxDD:  {hrp_report_ec.combined_result.max_drawdown:.2%}")
    except Exception as e:
        print(f"  HRP EC failed: {e}")
        hrp_report_ec = None

    # Step 5c: Risk Parity Portfolio on AG
    try:
        rp_report_ag = run_portfolio_backtest(kept, rp_w, TEST_SYMBOL, TEST_START, TEST_END, TOTAL_CAPITAL)
        print(f"\n  Risk Parity Portfolio AG:")
        print(f"    Sharpe: {rp_report_ag.combined_result.sharpe:.3f}")
        print(f"    Return: {rp_report_ag.combined_result.total_return:.2%}")
        print(f"    MaxDD:  {rp_report_ag.combined_result.max_drawdown:.2%}")
    except Exception as e:
        print(f"  RP AG failed: {e}")
        rp_report_ag = None

    # Save results
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save weights
    portfolio_result = {
        "method": "hrp",
        "n_strategies": len(kept),
        "sharpe_threshold": SHARPE_THRESHOLD,
        "corr_threshold": CORR_THRESHOLD,
        "total_capital": TOTAL_CAPITAL,
        "strategies": {},
    }
    for d in kept:
        ver = d["version"]
        portfolio_result["strategies"][ver] = {
            "weight_hrp": round(hrp_w.get(ver, 0), 4),
            "weight_rp": round(rp_w.get(ver, 0), 4),
            "mean_test_sharpe": round(d["mean_test"], 3),
            "ag_sharpe": round(d["ag_sharpe"] or 0, 3),
            "ec_sharpe": round(d["ec_sharpe"] or 0, 3),
            "freq": d["freq"],
        }

    if hrp_report_ag:
        portfolio_result["hrp_ag_sharpe"] = round(hrp_report_ag.combined_result.sharpe, 3)
        portfolio_result["hrp_ag_return"] = round(hrp_report_ag.combined_result.total_return, 4)
        portfolio_result["hrp_ag_maxdd"] = round(hrp_report_ag.combined_result.max_drawdown, 4)
    if hrp_report_ec:
        portfolio_result["hrp_ec_sharpe"] = round(hrp_report_ec.combined_result.sharpe, 3)
        portfolio_result["hrp_ec_return"] = round(hrp_report_ec.combined_result.total_return, 4)
        portfolio_result["hrp_ec_maxdd"] = round(hrp_report_ec.combined_result.max_drawdown, 4)
    if rp_report_ag:
        portfolio_result["rp_ag_sharpe"] = round(rp_report_ag.combined_result.sharpe, 3)
        portfolio_result["rp_ag_return"] = round(rp_report_ag.combined_result.total_return, 4)
        portfolio_result["rp_ag_maxdd"] = round(rp_report_ag.combined_result.max_drawdown, 4)

    # Correlation matrix
    if corr_ag is not None:
        portfolio_result["avg_correlation"] = round(
            corr_ag.values[np.triu_indices_from(corr_ag.values, k=1)].mean(), 3)

    weights_path = str(Path(OUTPUT_DIR) / "weights.json")
    with open(weights_path, "w") as f:
        json.dump(portfolio_result, f, indent=2)
    print(f"\nResults saved to {weights_path}")

    # Generate HTML reports
    reporter = HTMLReportGenerator()
    if hrp_report_ag:
        try:
            report_path = str(Path(QBASE_ROOT) / "reports" / "portfolio_hrp_ag.html")
            reporter.generate_portfolio_report(
                [s.result for s in hrp_report_ag.slot_results],
                hrp_report_ag.combined_result,
                report_path,
            )
            print(f"HTML report: {report_path}")
        except Exception as e:
            print(f"  HTML report failed: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*70}")
    print(f"  Strategies: {len(kept)}")
    if corr_ag is not None:
        print(f"  Avg correlation: {portfolio_result.get('avg_correlation', 'N/A')}")
    if hrp_report_ag:
        print(f"  HRP AG:  Sharpe={hrp_report_ag.combined_result.sharpe:.3f}  "
              f"Return={hrp_report_ag.combined_result.total_return:.2%}  "
              f"MaxDD={hrp_report_ag.combined_result.max_drawdown:.2%}")
    if hrp_report_ec:
        print(f"  HRP EC:  Sharpe={hrp_report_ec.combined_result.sharpe:.3f}  "
              f"Return={hrp_report_ec.combined_result.total_return:.2%}  "
              f"MaxDD={hrp_report_ec.combined_result.max_drawdown:.2%}")
    if rp_report_ag:
        print(f"  RP AG:   Sharpe={rp_report_ag.combined_result.sharpe:.3f}  "
              f"Return={rp_report_ag.combined_result.total_return:.2%}  "
              f"MaxDD={rp_report_ag.combined_result.max_drawdown:.2%}")

    # Best single strategy for comparison
    best_single = max(all_data, key=lambda x: x.get("ag_sharpe", 0) or 0)
    print(f"\n  Best single (AG): {best_single['version']} Sharpe={best_single['ag_sharpe']:.3f}")
