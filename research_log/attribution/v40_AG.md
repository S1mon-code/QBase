# strong_trend_v40 Attribution Report — AG (2025-01-01 ~ 2026-03-01)

## Signal Attribution (Ablation Test)

**Baseline Sharpe: 1.365** | Trades: 84

| Indicator | Role | Without It | Contribution | % of Total |
|-----------|------|:----------:|:------------:|:----------:|
| Kst Line | unknown | 1.071 | +0.295 | 21.6% |
| Kvo Sig | unknown | 1.219 | +0.146 | 10.7% |
| Psar Vals | unknown | 1.285 | +0.080 | 5.8% |
| Kvo Vals | unknown | 1.385 | -0.020 | -1.5% |
| Psar Dir | unknown | 1.454 | -0.089 | -6.5% |
| Kst Sig | unknown | 1.760 | -0.395 | -28.9% |

**Dominant indicator**: Kst Line
**Redundant indicators**: Kvo Vals

### Interpretation
- **Kst Sig** adds minimal value (-28.9%). Consider removing for simplicity.
- **Kvo Vals** adds minimal value (-1.5%). Consider removing for simplicity.
- **Psar Dir** adds minimal value (-6.5%). Consider removing for simplicity.

## Regime Attribution

Total round-trip trades analyzed: 43

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | 17 | 64.7% | +0.76% | +13.00% |
| Weak (15-25) | 21 | 42.9% | +0.22% | +4.70% |
| None (<15) | 5 | 80.0% | +4.34% | +21.71% |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| High (>75th) | 15 | 60.0% | +1.39% | +20.84% |
| Normal (25-75th) | 13 | 53.8% | +0.87% | +11.28% |
| Low (<25th) | 3 | 66.7% | +0.02% | +0.06% |
| Unknown | 12 | 50.0% | +0.60% | +7.22% |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | 4 | 100.0% | +2.84% | +11.37% |
| Normal (0.7-1.5x) | 24 | 54.2% | +1.10% | +26.40% |
| Quiet (<0.7x) | 15 | 46.7% | +0.11% | +1.63% |

### Cross Analysis: Trend x Volatility

| | High Vol | Normal Vol | Low Vol |
|--|:--:|:--:|:--:|
| **Strong Trend** | 58% (12) | 50% (2) | — |
| **Weak Trend** | 0% (1) | 44% (9) | 67% (3) |
| **No Trend** | 100% (2) | 100% (2) | — |

**Best regime**: trend=none
**Worst regime**: vol=low

- Strategy has strong regime dependency: win rate ranges from 43% to 80% across trend regimes.

---
*Generated: 2026-03-28 12:17*