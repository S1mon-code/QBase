# strong_trend_v20 Attribution Report — AG (2025-01-01 ~ 2026-03-01)

## Signal Attribution (Ablation Test)

**Baseline Sharpe: 1.804** | Trades: 18

| Indicator | Role | Without It | Contribution | % of Total |
|-----------|------|:----------:|:------------:|:----------:|
| Vroc | unknown | 1.121 | +0.683 | 37.8% |
| Frac High | unknown | 1.139 | +0.665 | 36.8% |
| Frac Low | unknown | 1.804 | +0.000 | 0.0% |
| Mi | unknown | 1.804 | -0.001 | -0.0% |

**Dominant indicator**: Vroc
**Redundant indicators**: Frac Low, Mi

### Interpretation
- **Frac Low** adds minimal value (0.0%). Consider removing for simplicity.
- **Mi** adds minimal value (-0.0%). Consider removing for simplicity.

## Regime Attribution

Total round-trip trades analyzed: 9

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | 8 | 87.5% | +4.40% | +35.18% |
| Weak (15-25) | 1 | 100.0% | +0.61% | +0.61% |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| High (>75th) | 5 | 100.0% | +4.17% | +20.86% |
| Unknown | 4 | 75.0% | +3.73% | +14.93% |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | 3 | 66.7% | +2.49% | +7.47% |
| Normal (0.7-1.5x) | 3 | 100.0% | +6.88% | +20.65% |
| Quiet (<0.7x) | 3 | 100.0% | +2.56% | +7.68% |

### Cross Analysis: Trend x Volatility

| | High Vol | Normal Vol | Low Vol |
|--|:--:|:--:|:--:|
| **Strong Trend** | 100% (5) | — | — |
| **Weak Trend** | — | — | — |
| **No Trend** | — | — | — |

**Best regime**: activity=normal
**Worst regime**: activity=active


---
*Generated: 2026-03-28 12:17*