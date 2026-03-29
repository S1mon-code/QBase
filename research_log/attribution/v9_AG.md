# strong_trend_v9 Attribution Report — AG (2025-01-01 ~ 2026-03-01)

## Signal Attribution (Ablation Test)

**Baseline Sharpe: 2.152** | Trades: 261

| Indicator | Role | Without It | Contribution | % of Total |
|-----------|------|:----------:|:------------:|:----------:|
| Fi | unknown | 1.482 | +0.670 | 31.1% |
| K Line | unknown | 1.706 | +0.446 | 20.7% |
| St Line | unknown | 2.131 | +0.021 | 1.0% |
| St Dir | unknown | 2.431 | -0.279 | -13.0% |

**Dominant indicator**: Fi
**Redundant indicators**: St Line

### Interpretation
- **St Dir** adds minimal value (-13.0%). Consider removing for simplicity.
- **St Line** adds minimal value (1.0%). Consider removing for simplicity.

## Regime Attribution

Total round-trip trades analyzed: 150

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | 66 | 50.0% | +0.51% | +33.87% |
| Weak (15-25) | 59 | 44.1% | +1.53% | +90.07% |
| None (<15) | 25 | 64.0% | +3.01% | +75.17% |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| High (>75th) | 58 | 50.0% | +0.84% | +48.46% |
| Normal (25-75th) | 54 | 57.4% | +1.52% | +82.10% |
| Low (<25th) | 25 | 52.0% | +2.92% | +73.07% |
| Unknown | 13 | 15.4% | -0.35% | -4.52% |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | 40 | 50.0% | +1.46% | +58.26% |
| Normal (0.7-1.5x) | 47 | 40.4% | +1.46% | +68.39% |
| Quiet (<0.7x) | 63 | 57.1% | +1.15% | +72.46% |

### Cross Analysis: Trend x Volatility

| | High Vol | Normal Vol | Low Vol |
|--|:--:|:--:|:--:|
| **Strong Trend** | 52% (29) | 67% (21) | 40% (5) |
| **Weak Trend** | 53% (19) | 50% (28) | 20% (10) |
| **No Trend** | 40% (10) | 60% (5) | 90% (10) |

**Best regime**: trend=none
**Worst regime**: vol=unknown


---
*Generated: 2026-03-28 12:17*