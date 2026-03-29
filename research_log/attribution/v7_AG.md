# strong_trend_v7 Attribution Report — AG (2025-01-01 ~ 2026-03-01)

## Signal Attribution (Ablation Test)

**Baseline Sharpe: 1.227** | Trades: 40

| Indicator | Role | Without It | Contribution | % of Total |
|-----------|------|:----------:|:------------:|:----------:|
| Psar Dir | unknown | 0.000 | +1.227 | 100.0% |
| Cmf | unknown | 1.193 | +0.034 | 2.8% |
| Psar Values | unknown | 1.227 | +0.000 | 0.0% |
| Tsi Line | unknown | 1.650 | -0.423 | -34.4% |
| Tsi Signal | unknown | 1.894 | -0.666 | -54.3% |

**Dominant indicator**: Psar Dir
**Redundant indicators**: Cmf, Psar Values

### Interpretation
- **Cmf** adds minimal value (2.8%). Consider removing for simplicity.
- Strategy is heavily dependent on **Psar Dir** (100% contribution). Consider if this is a feature or a risk.
- **Psar Values** adds minimal value (0.0%). Consider removing for simplicity.
- **Tsi Line** adds minimal value (-34.4%). Consider removing for simplicity.
- **Tsi Signal** adds minimal value (-54.3%). Consider removing for simplicity.

## Regime Attribution

Total round-trip trades analyzed: 20

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | 3 | 0.0% | -1.24% | -3.73% |
| Weak (15-25) | 8 | 75.0% | +1.57% | +12.59% |
| None (<15) | 9 | 88.9% | +2.11% | +18.96% |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| High (>75th) | 6 | 50.0% | +1.32% | +7.95% |
| Normal (25-75th) | 4 | 100.0% | +3.56% | +14.23% |
| Low (<25th) | 4 | 75.0% | +1.25% | +5.01% |
| Unknown | 6 | 66.7% | +0.10% | +0.63% |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | 4 | 0.0% | -1.16% | -4.62% |
| Normal (0.7-1.5x) | 7 | 85.7% | +2.04% | +14.27% |
| Quiet (<0.7x) | 9 | 88.9% | +2.02% | +18.17% |

### Cross Analysis: Trend x Volatility

| | High Vol | Normal Vol | Low Vol |
|--|:--:|:--:|:--:|
| **Strong Trend** | 0% (3) | — | — |
| **Weak Trend** | — | 100% (3) | 0% (1) |
| **No Trend** | 100% (3) | 100% (1) | 100% (3) |

**Best regime**: vol=normal
**Worst regime**: trend=strong

- Strategy has strong regime dependency: win rate ranges from 0% to 89% across trend regimes.

---
*Generated: 2026-03-28 12:17*