# Strong Trend Strategy — Complete Results

## Overview

- **目标**: 开发品种无关的强趋势策略，捕捉涨幅 >100%、持续 6+ 个月的大行情
- **策略数量**: 20 个 (v1-v20)，每个最多 3 个指标，只做多
- **训练集**: 16 个品种强趋势时段 (J, ZC, JM, LC, SF, I, SM, SA, IF, CF, NI, SC, B, P, RU, PS)
- **优化训练**: 6 个代表品种 (J, ZC, JM, I, NI, SA)，Optuna TPE 150-200 trials
- **测试集**: AG (2025-01 → 2026-03, +254%), EC (2023-07 → 2024-09, +907%)
- **结果**: 19/20 策略测试集平均 Sharpe > 0

## Test Set Results (AG + EC)

| Rank | Version | 策略名 | 指标组合 | 频率 | Train Sharpe | AG Sharpe | EC Sharpe | Mean Test |
|------|---------|--------|---------|------|-------------|-----------|-----------|-----------|
| 1 | v12 | Aroon动量 | Aroon + PPO + Volume Momentum | daily | 0.885 | 1.397 | 2.308 | **1.853** |
| 2 | v20 | 分形突破 | Fractal + Mass Index + VROC | daily | 0.971 | 1.329 | 2.213 | **1.771** |
| 3 | v11 | 涡旋指标 | Vortex + ROC + OI Momentum | daily | 0.945 | 0.680 | 2.833 | **1.757** |
| 4 | v9 | Supertrend振荡 | Supertrend + StochRSI + Force Index | 1h | 0.822 | 2.021 | 1.446 | **1.733** |
| 5 | v8 | 回归斜率 | LinReg Slope + Choppiness + NATR | daily | 0.764 | 1.314 | 2.133 | **1.724** |
| 6 | v18 | DEMA知而 | DEMA + KST + NR7 | daily | 0.557 | 0.815 | 2.550 | **1.683** |
| 7 | v16 | T3平滑 | T3 + Ergodic + Twiggs MF | 4h | 0.872 | 1.668 | 1.565 | **1.617** |
| 8 | v19 | McGinley趋势 | McGinley + Coppock + A/D Line | daily | 0.725 | 1.502 | 1.477 | **1.490** |
| 9 | v15 | EMA扇形 | EMA Ribbon + CMO + ATR | daily | 0.789 | 1.337 | 1.326 | **1.332** |
| 10 | v2 | 趋势质量 | KAMA + R² + ATR | daily | 0.663 | 1.118 | 1.465 | **1.292** |
| 11 | v7 | SAR动量 | PSAR + TSI + CMF | 4h | 0.680 | 1.278 | 1.274 | **1.276** |
| 12 | v10 | 三重EMA | TEMA + ADX + Bollinger Width | daily | 0.700 | 1.154 | 1.345 | **1.250** |
| 13 | v14 | 突破随机游走 | Donchian + RWI + Klinger | daily | 0.859 | 1.434 | 1.036 | **1.235** |
| 14 | v1 | Supertrend动量 | Supertrend + ROC + Volume Spike | daily | 0.545 | 0.955 | 1.456 | **1.205** |
| 15 | v6 | 一目均衡 | Ichimoku + RSI + OBV | daily | 0.819 | 1.841 | 0.278 | **1.060** |
| 16 | v13 | 零滞后趋势 | ZLEMA + Fisher + Range Expansion | 4h | 0.596 | 0.506 | 1.557 | **1.032** |
| 17 | v17 | ALMA趋势 | ALMA + Ultimate Osc + EMV | daily | 0.850 | 0.774 | 0.415 | **0.594** |
| 18 | v5 | Hull趋势 | HMA + CCI + Volume Climax | daily | 0.786 | 0.002 | 1.013 | **0.508** |
| 19 | v3 | 海龟突破 | Donchian + ADX + Chandelier | daily | 0.766 | -0.404 | 1.014 | **0.305** |
| 20 | v4 | EMA交叉 | EMA Cross + MACD + ATR | 4h | 0.262 | -1.258 | 0.000 | **-0.629** |

## Optimized Parameters (All 20 Strategies)

### v1 — Supertrend + ROC + Volume Spike (daily)
```json
{"st_period": 17, "st_mult": 3.536, "roc_period": 15, "roc_threshold": 4.126, "vol_threshold": 1.670}
```

### v2 — KAMA + R² + ATR (daily)
```json
{"kama_period": 15, "rsq_period": 32, "rsq_threshold": 0.499, "atr_period": 13, "trail_mult": 4.977}
```

### v3 — Donchian + ADX + Chandelier Exit (daily)
```json
{"don_period": 31, "adx_period": 20, "adx_threshold": 20.18, "chand_period": 15, "chand_mult": 3.146}
```

### v4 — EMA Cross + MACD + ATR (4h)
```json
{"ema_fast": 17, "ema_slow": 85, "macd_fast": 20, "macd_slow": 38, "atr_trail_mult": 2.682}
```

### v5 — HMA + CCI + Volume Climax (daily)
```json
{"hma_period": 10, "cci_period": 23, "cci_threshold": 50.47, "climax_period": 20, "atr_trail_mult": 4.299}
```

### v6 — Ichimoku + RSI + OBV (daily)
```json
{"tenkan": 12, "kijun": 16, "rsi_period": 19, "rsi_threshold": 40.02, "atr_trail_mult": 2.060}
```

### v7 — PSAR + TSI + CMF (4h)
```json
{"psar_af_step": 0.0367, "psar_af_max": 0.282, "tsi_long": 34, "tsi_short": 8, "atr_trail_mult": 4.877}
```

### v8 — LinReg Slope + Choppiness + NATR (daily)
```json
{"slope_period": 18, "chop_period": 19, "chop_threshold": 61.96, "natr_period": 25, "atr_trail_mult": 4.876}
```

### v9 — Supertrend + StochRSI + Force Index (1h)
```json
{"st_period": 17, "st_mult": 2.232, "stochrsi_period": 19, "fi_period": 8, "atr_trail_mult": 2.365}
```

### v10 — TEMA + ADX + Bollinger Width (daily)
```json
{"tema_period": 40, "adx_period": 11, "adx_threshold": 20.64, "bb_period": 20, "atr_trail_mult": 4.685}
```

### v11 — Vortex + ROC + OI Momentum (daily)
```json
{"vortex_period": 22, "roc_period": 14, "roc_threshold": 2.080, "oi_period": 29, "atr_trail_mult": 4.590}
```

### v12 — Aroon + PPO + Volume Momentum (daily)
```json
{"aroon_period": 21, "ppo_fast": 16, "ppo_slow": 24, "vol_mom_period": 24, "atr_trail_mult": 4.807}
```

### v13 — ZLEMA + Fisher Transform + Range Expansion (4h)
```json
{"zlema_period": 33, "fisher_period": 5, "re_period": 19, "re_threshold": 1.207, "atr_trail_mult": 3.346}
```

### v14 — Donchian + RWI + Klinger (daily)
```json
{"don_period": 40, "rwi_period": 20, "klinger_fast": 43, "klinger_slow": 40, "atr_trail_mult": 4.872}
```

### v15 — EMA Ribbon + CMO + ATR (daily)
```json
{"ribbon_base": 11, "cmo_period": 16, "cmo_threshold": 10.24, "atr_period": 12, "atr_trail_mult": 4.763}
```

### v16 — T3 + Ergodic + Twiggs MF (4h)
```json
{"t3_period": 4, "t3_vfactor": 0.658, "ergo_short": 3, "ergo_long": 17, "atr_trail_mult": 2.192}
```

### v17 — ALMA + Ultimate Oscillator + EMV (daily)
```json
{"alma_period": 7, "alma_offset": 0.931, "uo_p1": 6, "uo_p2": 11, "atr_trail_mult": 3.620}
```

### v18 — DEMA + KST + NR7 (daily)
```json
{"dema_period": 40, "kst_signal": 6, "nr7_lookback": 10, "atr_period": 24, "atr_trail_mult": 2.446}
```

### v19 — McGinley Dynamic + Coppock + A/D Line (daily)
```json
{"mcg_period": 8, "cop_wma": 10, "cop_roc_long": 13, "cop_roc_short": 13, "atr_trail_mult": 4.990}
```

### v20 — Fractal Levels + Mass Index + VROC (daily)
```json
{"fractal_period": 3, "mass_ema": 9, "mass_sum": 27, "vroc_period": 16, "atr_trail_mult": 4.830}
```

## Key Findings

### 1. 强趋势需要宽止损
大多数表现好的策略 `atr_trail_mult` 优化到 4.0-5.0。说明强趋势中回撤剧烈，窄止损会被频繁震出。

### 2. 量价指标加分明显
Top 5 中有 4 个使用了量/OI 相关指标 (Volume Momentum, VROC, OI Momentum, Force Index)。中国期货市场量价信号价值高。

### 3. 非传统指标表现优异
v12 (Aroon+PPO) 和 v20 (Fractal+Mass Index) 排名最高，都不是"经典"趋势策略组合。传统的 v3 (海龟) 反而排名第 19。

### 4. 频率不是决定因素
daily (14个), 4h (4个), 1h (1个) 都有表现好的策略。v9 (1h) 在 AG 上 Sharpe 达到 2.02。

### 5. v4 (EMA+MACD 4h) 唯一失败
经过 3 轮迭代优化仍为负。EMA 交叉在强趋势中信号太慢，且回撤时容易反复交叉产生损耗。

## Training Set Details

| Symbol | Period | Rally | Sector |
|--------|--------|-------|--------|
| J | 2015-06 → 2017-06 | +210% | 黑色系 (供给侧改革) |
| ZC | 2020-06 → 2022-06 | +253% | 能源 (煤炭供需错配) |
| JM | 2020-06 → 2022-06 | +170% | 黑色系 (碳中和) |
| I | 2015-06 → 2017-06 | +134% | 黑色系 (供给侧改革) |
| NI | 2021-01 → 2022-09 | +106% | 有色 (伦镍逼空) |
| SA | 2022-06 → 2024-06 | +130% | 化工 (光伏需求) |
