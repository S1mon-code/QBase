# strong_trend_v12 Attribution Report — AG (2025-01-01 ~ 2026-03-01)

## Signal Attribution (Ablation Test)

**Baseline Sharpe: 3.090** | Trades: 18

| Indicator | Role | Without It | Contribution | % of Total |
|-----------|------|:----------:|:------------:|:----------:|
| Volume Momentum | volume | 1.494 | +1.597 | 51.7% |
| PPO Histogram | momentum | 3.004 | +0.086 | 2.8% |
| Aroon Oscillator | trend | 3.087 | +0.003 | 0.1% |

**Dominant indicator**: Volume Momentum
**Redundant indicators**: Aroon Oscillator, PPO Histogram

### Interpretation
- **Aroon Oscillator** adds minimal value (0.1%). Consider removing for simplicity.
- **PPO Histogram** adds minimal value (2.8%). Consider removing for simplicity.

## Regime Attribution

Total round-trip trades analyzed: 9

### By Trend Strength (ADX)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Strong (ADX>25) | 3 | 100.0% | +173.26% | +519.79% |
| Weak (15-25) | 3 | 0.0% | -6.76% | -20.29% |
| None (<15) | 3 | 100.0% | +5.18% | +15.55% |

### By Volatility (ATR Percentile)

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Unknown | 9 | 66.7% | +57.23% | +515.04% |

### By Volume Activity

| Regime | Trades | Win Rate | Avg PnL | Total PnL |
|--------|:------:|:--------:|:-------:|:---------:|
| Active (>1.5x) | 3 | 100.0% | +173.26% | +519.79% |
| Normal (0.7-1.5x) | 5 | 60.0% | -0.69% | -3.45% |
| Quiet (<0.7x) | 1 | 0.0% | -1.30% | -1.30% |

**Best regime**: trend=strong
**Worst regime**: trend=weak

- Strategy has strong regime dependency: win rate ranges from 0% to 100% across trend regimes.

---
*Generated: 2026-03-28 11:05*