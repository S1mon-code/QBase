# Strong Trend Strategy — Optimization & Validation Results

## Test Set: AG (2025-01 → 2026-03), EC (2023-07 → 2024-09)


| Rank | Strategy | Train Sharpe | AG Sharpe | EC Sharpe | Mean Test |
|------|----------|-------------|-----------|-----------|-----------|
| 1 | v12 | 0.885 | 1.397 | 2.308 | 1.853 |
| 2 | v20 | 0.971 | 1.329 | 2.213 | 1.771 |
| 3 | v11 | 0.945 | 0.680 | 2.833 | 1.757 |
| 4 | v9 | 0.822 | 2.021 | 1.446 | 1.733 |
| 5 | v8 | 0.764 | 1.314 | 2.133 | 1.724 |
| 6 | v18 | 0.557 | 0.815 | 2.550 | 1.683 |
| 7 | v16 | 0.872 | 1.668 | 1.565 | 1.617 |
| 8 | v19 | 0.725 | 1.502 | 1.477 | 1.490 |
| 9 | v15 | 0.789 | 1.337 | 1.326 | 1.332 |
| 10 | v2 | 0.663 | 1.118 | 1.465 | 1.292 |
| 11 | v7 | 0.680 | 1.278 | 1.274 | 1.276 |
| 12 | v10 | 0.700 | 1.154 | 1.345 | 1.250 |
| 13 | v14 | 0.859 | 1.434 | 1.036 | 1.235 |
| 14 | v1 | 0.545 | 0.955 | 1.456 | 1.205 |
| 15 | v6 | 0.819 | 1.841 | 0.278 | 1.060 |
| 16 | v13 | 0.596 | 0.506 | 1.557 | 1.032 |
| 17 | v17 | 0.850 | 0.774 | 0.415 | 0.594 |
| 18 | v5 | 0.786 | 0.002 | 1.013 | 0.508 |
| 19 | v3 | 0.766 | -0.404 | 1.014 | 0.305 |
| 20 | v4 | 0.262 | -1.258 | 0.000 | -0.629 |

## Best Parameters (Top 5)

### v12
- Train Sharpe: 0.885
- Test AG: 1.397
- Test EC: 2.308
- Params: `{"aroon_period": 21, "ppo_fast": 16, "ppo_slow": 24, "vol_mom_period": 24, "atr_trail_mult": 4.806977702234153}`

### v20
- Train Sharpe: 0.971
- Test AG: 1.329
- Test EC: 2.213
- Params: `{"fractal_period": 3, "mass_ema": 9, "mass_sum": 27, "vroc_period": 16, "atr_trail_mult": 4.830105859193188}`

### v11
- Train Sharpe: 0.945
- Test AG: 0.680
- Test EC: 2.833
- Params: `{"vortex_period": 22, "roc_period": 14, "roc_threshold": 2.0797495833527, "oi_period": 29, "atr_trail_mult": 4.589840967575952}`

### v9
- Train Sharpe: 0.822
- Test AG: 2.021
- Test EC: 1.446
- Params: `{"st_period": 17, "st_mult": 2.231831989921357, "stochrsi_period": 19, "fi_period": 8, "atr_trail_mult": 2.364819129913487}`

### v8
- Train Sharpe: 0.764
- Test AG: 1.314
- Test EC: 2.133
- Params: `{"slope_period": 18, "chop_period": 19, "chop_threshold": 61.96283707038288, "natr_period": 25, "atr_trail_mult": 4.875727873989262}`
