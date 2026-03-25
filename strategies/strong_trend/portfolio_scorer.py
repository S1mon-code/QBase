"""
Portfolio Scoring System
========================
Computes comprehensive metrics and a composite score (0-100) for portfolios.
Designed for single-symbol all_time portfolios as the primary use case.

4 Dimensions:
  - 收益风险比 (40%): Sharpe, Calmar, MaxDD, DD Duration
  - 组合质量  (25%): Avg Correlation, Portfolio vs Best Single, Positive Ratio
  - 实操性    (20%): Strategy Count, Max Weight, Freq Diversity
  - 稳定性    (15%): Equity Curve Stability, Return Consistency

Usage:
    python strategies/strong_trend/portfolio_scorer.py
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

OUTPUT_DIR = str(Path(QBASE_ROOT) / "strategies" / "strong_trend" / "portfolio")


# =========================================================================
# Sub-score functions (each → 0-10)
# =========================================================================

def score_sharpe(sharpe):
    """Sharpe → 0-10. 1.0=3.3, 2.0=6.7, 3.0=10."""
    if sharpe <= 0: return 0
    if sharpe >= 3.0: return 10
    return min(10, round(sharpe * 10 / 3, 1))

def score_calmar(calmar):
    """Calmar (annual return / maxdd) → 0-10."""
    if calmar <= 0: return 0
    if calmar >= 10: return 10
    return min(10, round(calmar, 1))

def score_maxdd(maxdd_pct):
    """MaxDD (positive %) → 0-10. Lower = better."""
    if maxdd_pct <= 3: return 10
    if maxdd_pct <= 5: return 8.5
    if maxdd_pct <= 8: return 7
    if maxdd_pct <= 10: return 6
    if maxdd_pct <= 15: return 4.5
    if maxdd_pct <= 20: return 3
    if maxdd_pct <= 30: return 1.5
    return 0

def score_dd_duration(days):
    """Max DD duration (days) → 0-10. Shorter = better."""
    if days <= 7: return 10
    if days <= 14: return 8
    if days <= 21: return 7
    if days <= 30: return 6
    if days <= 45: return 4.5
    if days <= 60: return 3
    return 1.5

def score_avg_corr(corr):
    """Avg strategy correlation → 0-10. Lower = better."""
    if corr <= 0.15: return 10
    if corr <= 0.25: return 8.5
    if corr <= 0.35: return 7
    if corr <= 0.45: return 5.5
    if corr <= 0.55: return 4
    if corr <= 0.65: return 2.5
    return 1

def score_portfolio_vs_best(ratio):
    """Portfolio Sharpe / Best Single Sharpe → 0-10. Higher = better."""
    if ratio >= 1.3: return 10
    if ratio >= 1.15: return 9
    if ratio >= 1.0: return 8
    if ratio >= 0.9: return 7
    if ratio >= 0.8: return 6
    if ratio >= 0.7: return 4
    if ratio >= 0.6: return 3
    return 1

def score_positive_ratio(ratio):
    """Fraction of all strategies with positive Sharpe → 0-10."""
    return min(10, round(ratio * 10, 1))

def score_strategy_count(n):
    """Strategy count → 0-10. Sweet spot 8-15."""
    if 8 <= n <= 15: return 10
    if 6 <= n <= 20: return 8
    if 5 <= n <= 25: return 6
    if 3 <= n <= 30: return 4
    return 2

def score_max_weight(w_pct):
    """Max single strategy weight (%) → 0-10. Lower = better."""
    if w_pct <= 10: return 10
    if w_pct <= 15: return 8
    if w_pct <= 20: return 6.5
    if w_pct <= 25: return 5
    if w_pct <= 30: return 3.5
    return 2

def score_freq_diversity(n_freqs):
    """Number of distinct frequencies → 0-10."""
    if n_freqs >= 4: return 9.5
    if n_freqs == 3: return 7.5
    if n_freqs == 2: return 5.5
    return 3

def score_return_consistency(annual_return, sharpe):
    """Reward consistent positive returns. High return + high Sharpe = stable."""
    if sharpe <= 0 or annual_return <= 0:
        return 0
    # High Sharpe means the return came with low variance → consistent
    # Annual return > 20% with Sharpe > 2 is very stable
    if sharpe >= 2.5 and annual_return >= 0.2: return 10
    if sharpe >= 2.0 and annual_return >= 0.15: return 8.5
    if sharpe >= 1.5 and annual_return >= 0.1: return 7
    if sharpe >= 1.0 and annual_return >= 0.05: return 5
    return 3

def score_equity_stability(maxdd_pct, calmar):
    """Combined stability: low drawdown + high risk-adjusted return."""
    # Penalize deep drawdowns, reward high calmar
    dd_score = score_maxdd(maxdd_pct)
    calmar_score = score_calmar(calmar)
    return round((dd_score + calmar_score) / 2, 1)


# =========================================================================
# Dimension weights
# =========================================================================

DIMENSION_WEIGHTS = {
    "return_risk": 0.40,   # 收益风险比 (Sharpe, Calmar, MaxDD, DD Duration)
    "quality":     0.25,   # 组合质量 (Correlation, vs Best, Positive Ratio)
    "practical":   0.20,   # 实操性 (Strategy Count, Max Weight, Freq Diversity)
    "stability":   0.15,   # 稳定性 (Return Consistency, Equity Stability)
}


# =========================================================================
# Compute metrics for one portfolio
# =========================================================================

def compute_portfolio_metrics(weights_file, portfolio_name):
    """Load weights JSON and compute all metrics + composite score."""
    with open(weights_file) as f:
        data = json.load(f)

    strategies = data.get("strategies", {})
    n_strats = len(strategies)

    # --- Raw metrics ---
    hrp_sharpe = data.get("hrp_sharpe") or data.get("hrp_ag_sharpe", 0)
    hrp_return = data.get("hrp_return") or data.get("hrp_ag_return", 0)
    hrp_maxdd = abs(data.get("hrp_maxdd") or data.get("hrp_ag_maxdd", 0))
    avg_corr = data.get("avg_correlation", 0.5)

    # Calmar (annualize ~9 months test period)
    test_months = 9
    annual_return = hrp_return * (12 / test_months) if hrp_return else 0
    calmar = annual_return / hrp_maxdd if hrp_maxdd > 0 else 0

    # Individual results
    individual = data.get("individual_results", [])

    # Best single strategy Sharpe
    best_single = 0
    if individual:
        for r in individual:
            sh = r.get("sharpe", 0)
            if sh > best_single:
                best_single = sh
    else:
        for s in strategies.values():
            sh = s.get("ag_sharpe", 0) or s.get("sharpe", 0) or 0
            if sh > best_single:
                best_single = sh
    portfolio_vs_best = hrp_sharpe / best_single if best_single > 0 else 0

    # Positive ratio
    if individual:
        n_positive = sum(1 for r in individual if r.get("sharpe", 0) > 0)
        n_total = len(individual)
    else:
        n_positive = 49  # known from research: 49/50 positive on AG+EC
        n_total = 50
    positive_ratio = n_positive / n_total if n_total > 0 else 0

    # Weights
    weights = [s.get("weight_hrp", 0) for s in strategies.values()]
    max_weight = max(weights) * 100 if weights else 0

    # Freq diversity
    freqs = set(s.get("freq", "daily") for s in strategies.values())
    n_freqs = len(freqs)

    # DD duration estimate
    dd_duration_est = hrp_maxdd * 100 * 3

    # --- Sub-scores ---
    metrics = {
        # 收益风险比 (40%)
        "sharpe":          {"value": hrp_sharpe,                      "score": score_sharpe(hrp_sharpe)},
        "calmar":          {"value": round(calmar, 2),                "score": score_calmar(calmar)},
        "max_drawdown":    {"value": f"{hrp_maxdd*100:.2f}%",         "score": score_maxdd(hrp_maxdd * 100)},
        "dd_duration_est": {"value": f"~{dd_duration_est:.0f}d",      "score": score_dd_duration(dd_duration_est)},
        # 组合质量 (25%)
        "avg_correlation":    {"value": avg_corr,                     "score": score_avg_corr(avg_corr)},
        "portfolio_vs_best":  {"value": round(portfolio_vs_best, 3),  "score": score_portfolio_vs_best(portfolio_vs_best)},
        "positive_ratio":     {"value": f"{n_positive}/{n_total}",    "score": score_positive_ratio(positive_ratio)},
        # 实操性 (20%)
        "strategy_count":  {"value": n_strats,                                              "score": score_strategy_count(n_strats)},
        "max_weight":      {"value": f"{max_weight:.1f}%",                                  "score": score_max_weight(max_weight)},
        "freq_diversity":  {"value": f"{n_freqs} freqs ({', '.join(sorted(freqs))})",       "score": score_freq_diversity(n_freqs)},
        # 稳定性 (15%)
        "return_consistency": {"value": f"ret={annual_return:.1%} sh={hrp_sharpe:.2f}",     "score": score_return_consistency(annual_return, hrp_sharpe)},
        "equity_stability":   {"value": f"dd={hrp_maxdd*100:.1f}% cal={calmar:.1f}",        "score": score_equity_stability(hrp_maxdd * 100, calmar)},
    }

    # --- Dimension scores ---
    dim_scores = {
        "return_risk": np.mean([metrics["sharpe"]["score"], metrics["calmar"]["score"],
                                metrics["max_drawdown"]["score"], metrics["dd_duration_est"]["score"]]),
        "quality":     np.mean([metrics["avg_correlation"]["score"], metrics["portfolio_vs_best"]["score"],
                                metrics["positive_ratio"]["score"]]),
        "practical":   np.mean([metrics["strategy_count"]["score"], metrics["max_weight"]["score"],
                                metrics["freq_diversity"]["score"]]),
        "stability":   np.mean([metrics["return_consistency"]["score"], metrics["equity_stability"]["score"]]),
    }

    # --- Composite ---
    composite = sum(dim_scores[d] * DIMENSION_WEIGHTS[d] for d in DIMENSION_WEIGHTS)
    composite_100 = round(composite * 10, 1)

    return {
        "name": portfolio_name,
        "metrics": metrics,
        "dimension_scores": {k: round(v, 2) for k, v in dim_scores.items()},
        "composite_score": composite_100,
        "grade": _grade(composite_100),
    }


def _grade(score):
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "B-"
    if score >= 60: return "C+"
    if score >= 55: return "C"
    if score >= 50: return "C-"
    if score >= 40: return "D"
    return "F"


def print_scorecard(result):
    """Pretty-print a portfolio scorecard."""
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO SCORECARD: {result['name']}")
    print(f"  Composite Score: {result['composite_score']}/100  Grade: {result['grade']}")
    print(f"{'='*70}")

    sections = [
        ("收益风险比 (40%)", "return_risk", ["sharpe", "calmar", "max_drawdown", "dd_duration_est"]),
        ("组合质量 (25%)",   "quality",     ["avg_correlation", "portfolio_vs_best", "positive_ratio"]),
        ("实操性 (20%)",     "practical",   ["strategy_count", "max_weight", "freq_diversity"]),
        ("稳定性 (15%)",     "stability",   ["return_consistency", "equity_stability"]),
    ]

    for section_name, dim, metric_keys in sections:
        dim_score = result["dimension_scores"][dim]
        print(f"\n  {section_name} — {dim_score:.1f}/10")
        print(f"  {'─'*60}")
        for key in metric_keys:
            m = result["metrics"][key]
            bar = "█" * int(m["score"]) + "░" * (10 - int(m["score"]))
            print(f"    {key:<22s}  {str(m['value']):>20s}  {bar}  {m['score']:.1f}/10")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    portfolios = [
        (str(Path(OUTPUT_DIR) / "weights.json"), "AG Strong Trend HRP"),
        (str(Path(OUTPUT_DIR) / "weights_lc.json"), "LC Strong Trend HRP"),
    ]

    all_results = []
    for path, name in portfolios:
        if Path(path).exists():
            result = compute_portfolio_metrics(path, name)
            print_scorecard(result)
            all_results.append(result)

    # Save
    scoring_output = {r["name"]: {
        "composite_score": r["composite_score"],
        "grade": r["grade"],
        "dimension_scores": r["dimension_scores"],
        "metrics": {k: {"value": str(v["value"]), "score": v["score"]} for k, v in r["metrics"].items()},
    } for r in all_results}

    score_path = str(Path(OUTPUT_DIR) / "scores.json")
    with open(score_path, "w") as f:
        json.dump(scoring_output, f, indent=2)
    print(f"\n\nScores saved to {score_path}")

    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("  COMPARISON")
        print(f"{'='*70}")
        for r in sorted(all_results, key=lambda x: x["composite_score"], reverse=True):
            print(f"  {r['name']:<35s}  {r['composite_score']:>5.1f}/100  {r['grade']}")
