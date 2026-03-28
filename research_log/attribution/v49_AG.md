# strong_trend_v49 Attribution Report — AG (2025-01-01 ~ 2026-03-01)

## Signal Attribution (Ablation Test)

**Baseline Sharpe: 1.763** | Trades: 26

| Indicator | Role | Without It | Contribution | % of Total |
|-----------|------|:----------:|:------------:|:----------:|
| Vol Spikes | unknown | 0.000 | +1.763 | 100.0% |
| Hurst | unknown | 1.101 | +0.662 | 37.5% |
| St Dir | unknown | 1.780 | -0.017 | -1.0% |
| St Line | unknown | 1.828 | -0.065 | -3.7% |

**Dominant indicator**: Vol Spikes
**Redundant indicators**: St Dir, St Line

### Interpretation
- **St Dir** adds minimal value (-1.0%). Consider removing for simplicity.
- **St Line** adds minimal value (-3.7%). Consider removing for simplicity.
- Strategy is heavily dependent on **Vol Spikes** (100% contribution). Consider if this is a feature or a risk.

## Regime Attribution

Total round-trip trades analyzed: 13

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | 10 | 50.0% | +7.15% | +71.55% |
| Weak (15-25) | 3 | 66.7% | -1.83% | -5.49% |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| High (>75th) | 4 | 75.0% | +18.46% | +73.84% |
| Unknown | 9 | 44.4% | -0.87% | -7.79% |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | 10 | 50.0% | +6.35% | +63.54% |
| Normal (0.7-1.5x) | 2 | 50.0% | +1.10% | +2.21% |
| Quiet (<0.7x) | 1 | 100.0% | +0.30% | +0.30% |

### Cross Analysis: Trend x Volatility

| | High Vol | Normal Vol | Low Vol |
|--|:--:|:--:|:--:|
| **Strong Trend** | 75% (4) | — | — |
| **Weak Trend** | — | — | — |
| **No Trend** | — | — | — |

**Best regime**: vol=high
**Worst regime**: trend=weak


---
*Generated: 2026-03-28 12:17*