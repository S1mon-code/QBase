# QBase 指标库（320 个）

10 大类 320 个指标，纯 numpy 函数（numpy in → numpy out）。所有指标位于 `indicators/` 目录下对应分类文件夹中。

---

## Momentum — 动量/振荡类（35个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Rate of Change | `roc.py` | `rate_of_change(closes, period=12)` |
| RSI | `rsi.py` | `rsi(closes, period=14)` |
| MACD | `macd.py` | `macd(closes, fast=12, slow=26, signal=9)` → (line, signal, hist) |
| CCI | `cci.py` | `cci(highs, lows, closes, period=20)` |
| Stochastic | `stochastic.py` | `stochastic(highs, lows, closes, k=14, d=3)` → (%K, %D) |
| KDJ | `stochastic.py` | `kdj(highs, lows, closes, period=9, k=3, d=3)` → (K, D, J) |
| Williams %R | `williams_r.py` | `williams_r(highs, lows, closes, period=14)` |
| KAMA | `kama.py` | `kama(closes, period=10, fast_sc=2, slow_sc=30)` |
| TSMOM | `tsmom.py` | `tsmom(closes, lookback=252, vol_lookback=60)` |
| Momentum Accel | `momentum_accel.py` | `momentum_acceleration(closes, fast=10, slow=20)` |
| Awesome Oscillator | `awesome_oscillator.py` | `ao(highs, lows, fast=5, slow=34)` |
| Chande MO | `cmo.py` | `cmo(closes, period=14)` |
| Detrended Price | `dpo.py` | `dpo(closes, period=20)` |
| PPO | `ppo.py` | `ppo(closes, fast=12, slow=26, signal=9)` → (ppo, signal, hist) |
| Ultimate Osc | `ultimate_oscillator.py` | `ultimate_oscillator(highs, lows, closes, 7, 14, 28)` |
| Coppock Curve | `coppock.py` | `coppock(closes, wma=10, roc_long=14, roc_short=11)` |
| TSI | `tsi.py` | `tsi(closes, long=25, short=13, signal=7)` → (tsi, signal) |
| KST | `kst.py` | `kst(closes, ...)` → (kst, signal) |
| RVI | `rvi.py` | `rvi(opens, highs, lows, closes, period=10)` → (rvi, signal) |
| Elder Force | `elder_force.py` | `elder_force_index(closes, volumes, period=13)` |
| Schaff Trend | `schaff_trend.py` | `schaff_trend_cycle(closes, period=10, fast=23, slow=50)` |
| Connors RSI | `connors_rsi.py` | `connors_rsi(closes, rsi=3, streak=2, pct_rank=100)` |
| Fisher Transform | `fisher_transform.py` | `fisher_transform(highs, lows, period=10)` → (fisher, trigger) |
| Stochastic RSI | `stoch_rsi.py` | `stoch_rsi(closes, rsi=14, stoch=14, k=3, d=3)` → (%K, %D) |
| Choppiness Index | `chop_zone.py` | `choppiness_index(highs, lows, closes, period=14)` |
| Ergodic | `ergodic.py` | `ergodic(closes, short=5, long=20, signal=5)` → (erg, signal) |
| Price Momentum Quality | `price_momentum_quality.py` | `pmq(closes, period=20)` |
| Acceleration Bands | `acceleration_bands.py` | `acceleration_bands(highs, lows, closes, period=20, width=0.04)` |
| Relative Vigor Index | `relative_vigor.py` | `relative_vigor_index(opens, highs, lows, closes, period=10)` |
| Pretty Good Oscillator | `pretty_good_oscillator.py` | `pretty_good_oscillator(closes, highs, lows, period=14)` |
| Center of Gravity | `center_of_gravity.py` | `cog(closes, period=10)` |
| Cyber Cycle | `cyber_cycle.py` | `cyber_cycle(closes, alpha=0.07)` |
| Reflex | `reflex.py` | `reflex(closes, period=20)` |
| TrendFlex | `trend_flex.py` | `trendflex(closes, period=20)` |
| Rocket RSI | `rocket_rsi.py` | `rocket_rsi(closes, rsi_period=10, rocket_period=8)` |
| Relative Momentum Index | `relative_momentum_index.py` | `rmi(closes, period=14, lookback=5)` |

## Trend — 趋势类（35个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| ADX | `adx.py` | `adx(highs, lows, closes, period=14)` |
| Supertrend | `supertrend.py` | `supertrend(highs, lows, closes, period=10, mult=3.0)` → (line, dir) |
| Aroon | `aroon.py` | `aroon(highs, lows, period=25)` → (up, down, osc) |
| EMA | `ema.py` | `ema(data, period)` / `ema_cross(data, fast, slow)` |
| SMA | `sma.py` | `sma(data, period)` |
| Donchian | `donchian.py` | `donchian(highs, lows, period=20)` → (upper, lower, mid) |
| Keltner | `keltner.py` | `keltner(highs, lows, closes, ema=20, atr=10, mult=1.5)` → (u, m, l) |
| EMA Ribbon | `ema_ribbon.py` | `ema_ribbon(data, periods)` / `ema_ribbon_signal(data, periods)` |
| Fractal | `fractal.py` | `fractal_high(highs)` / `fractal_low(lows)` / `fractal_levels(h, l)` |
| Higher Low | `higher_low.py` | `higher_lows(lows, lookback=20)` / `lower_highs(highs, lookback=20)` |
| Ichimoku | `ichimoku.py` | `ichimoku(highs, lows, closes, 9, 26, 52, 26)` → 5 components |
| Parabolic SAR | `psar.py` | `psar(highs, lows, af_start=0.02, af_step=0.02, af_max=0.2)` → (sar, dir) |
| DEMA | `dema.py` | `dema(data, period)` |
| TEMA | `tema.py` | `tema(data, period)` |
| HMA | `hma.py` | `hma(data, period)` |
| VWMA | `vwma.py` | `vwma(closes, volumes, period=20)` |
| Linear Regression | `linear_regression.py` | `linear_regression(data, period)` / `linear_regression_slope()` / `r_squared()` |
| ALMA | `alma.py` | `alma(data, period=9, offset=0.85, sigma=6)` |
| T3 | `t3.py` | `t3(data, period=5, volume_factor=0.7)` |
| McGinley | `mcginley.py` | `mcginley_dynamic(data, period=14)` |
| Vortex | `vortex.py` | `vortex(highs, lows, closes, period=14)` → (VI+, VI-) |
| Mass Index | `mass_index.py` | `mass_index(highs, lows, ema=9, sum=25)` |
| RWI | `rwi.py` | `rwi(highs, lows, closes, period=14)` → (rwi_high, rwi_low) |
| ZLEMA | `zlema.py` | `zlema(data, period)` |
| Trend Intensity | `trend_intensity.py` | `trend_intensity(closes, period=14)` |
| Ehlers Instantaneous | `ehlers_instantaneous.py` | `instantaneous_trendline(closes, alpha=0.07)` |
| VIDYA | `vidya.py` | `vidya(closes, period=14, cmo_period=9)` |
| FRAMA | `frama.py` | `frama(closes, period=16)` |
| Jurik MA | `jurik_ma.py` | `jma(closes, period=7, phase=0.0, power=2.0)` |
| Laguerre Filter | `laguerre_filter.py` | `laguerre(closes, gamma=0.8)` |
| Kaufman ER Bands | `kaufman_er_bands.py` | `er_bands(closes, period=20, mult=1.0)` |
| Sine Wave | `sine_wave.py` | `ehlers_sine_wave(closes, alpha=0.07)` |
| Decycler | `decycler.py` | `decycler(closes, period=60)` |
| MESA Adaptive MA | `mesa_adaptive_ma.py` | `mama(closes, fast_limit=0.5, slow_limit=0.05)` |
| TRIX | `trix.py` | `trix(closes, period=15)` |

## Volatility — 波动率类（35个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| ATR | `atr.py` | `atr(highs, lows, closes, period=14)` |
| Bollinger Bands | `bollinger.py` | `bollinger_bands(closes, period=20, std=2.0)` → (u, m, l) |
| Historical Vol | `historical_vol.py` | `historical_volatility(closes, period=20, ann=252)` |
| TTM Squeeze | `ttm_squeeze.py` | `ttm_squeeze(h, l, c, bb=20, bb_m=2.0, kc=20, kc_m=1.5)` → (squeeze, mom) |
| Hurst | `hurst.py` | `hurst_exponent(data, max_lag=20)` |
| Entropy | `entropy.py` | `entropy(closes, period=20, bins=10)` |
| VoV | `vov.py` | `vov(closes, vol_period=20, vov_period=20)` |
| NR7 | `nr7.py` | `nr7(highs, lows)` / `nr4(highs, lows)` |
| Range Expansion | `range_expansion.py` | `range_expansion(highs, lows, closes, period=14)` |
| NATR | `natr.py` | `natr(highs, lows, closes, period=14)` |
| Chaikin Volatility | `chaikin_vol.py` | `chaikin_volatility(highs, lows, ema=10, roc=10)` |
| Ulcer Index | `ulcer_index.py` | `ulcer_index(closes, period=14)` |
| Garman-Klass | `garman_klass.py` | `garman_klass(opens, highs, lows, closes, period=20)` |
| Parkinson | `parkinson.py` | `parkinson(highs, lows, period=20)` |
| Rogers-Satchell | `rogers_satchell.py` | `rogers_satchell(opens, highs, lows, closes, period=20)` |
| Yang-Zhang | `yang_zhang.py` | `yang_zhang(opens, highs, lows, closes, period=20)` |
| Realized Variance | `realized_variance.py` | `realized_variance(closes, period=20)` / `realized_volatility(...)` |
| ADR | `adr.py` | `average_day_range(highs, lows, period=14)` / `adr_percent(...)` |
| Rolling Std | `std_dev.py` | `rolling_std(data, period=20)` / `z_score(data, period=20)` |
| True Range | `true_range.py` | `true_range(highs, lows, closes)` |
| Chandelier Exit | `chandelier_exit.py` | `chandelier_exit(highs, lows, closes, period=22, mult=3.0)` → (long, short) |
| Keltner Width | `keltner_width.py` | `keltner_width(highs, lows, closes, ema=20, atr=10, mult=1.5)` |
| Volatility Ratio | `vol_ratio.py` | `volatility_ratio(highs, lows, closes, period=14)` |
| ATR Ratio | `chop.py` | `atr_ratio(highs, lows, closes, short=5, long=20)` |
| Close-to-Close Vol | `close_to_close_vol.py` | `close_to_close_vol(closes, period=20, ann=252)` |
| Relative Volatility | `relative_vol.py` | `relative_volatility(closes, fast=10, slow=60)` |
| Intraday Intensity | `intraday_intensity.py` | `intraday_intensity(highs, lows, closes, volumes, period=20)` |
| Historical Skewness | `historical_skew.py` | `rolling_skewness(returns_or_closes, period=60)` |
| Historical Kurtosis | `historical_kurtosis.py` | `rolling_kurtosis(returns_or_closes, period=60)` |
| GK Volatility Ratio | `gk_volatility_ratio.py` | `gk_vol_ratio(opens, highs, lows, closes, fast=10, slow=60)` |
| Normalized Range | `normalized_range.py` | `normalized_range(highs, lows, closes, period=20)` |
| Conditional Volatility | `conditional_vol.py` | `conditional_volatility(closes, period=20, threshold=0.0)` |
| Vol-of-Vol Regime | `vol_of_vol_regime.py` | `vol_of_vol_regime(closes, vol_period=20, vov_period=20)` |
| Realized Skew | `realized_skew.py` | `realized_skewness(highs, lows, closes, period=20)` |
| Price Acceleration | `price_acceleration.py` | `price_acceleration(closes, period=14)` |

## Volume — 成交量/持仓类（39个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| OBV | `obv.py` | `obv(closes, volumes)` |
| MFI | `mfi.py` | `mfi(highs, lows, closes, volumes, period=14)` |
| VWAP | `vwap.py` | `vwap(highs, lows, closes, volumes)` |
| OI Divergence | `oi_divergence.py` | `oi_divergence(closes, oi, period=20)` |
| OI Momentum | `oi_momentum.py` | `oi_momentum(oi, period=20)` / `oi_sentiment(c, oi, v, period=20)` |
| Volume Spike | `volume_spike.py` | `volume_spike(v, period=20, threshold=2.0)` / `volume_climax(...)` / `volume_dry_up(...)` |
| Close Location | `close_location.py` | `close_location(highs, lows, closes)` |
| Price Density | `price_density.py` | `price_density(highs, lows, closes, period=20)` |
| VPT | `volume_price_trend.py` | `vpt(closes, volumes)` |
| A/D Line | `ad_line.py` | `ad_line(highs, lows, closes, volumes)` |
| CMF | `cmf.py` | `cmf(highs, lows, closes, volumes, period=20)` |
| EMV | `emv.py` | `emv(highs, lows, volumes, period=14)` |
| Force Index | `force_index.py` | `force_index(closes, volumes, period=13)` |
| Klinger | `klinger.py` | `klinger(highs, lows, closes, volumes, fast=34, slow=55, signal=13)` → (kvo, signal) |
| NVI/PVI | `nvi.py` | `nvi(closes, volumes)` / `pvi(closes, volumes)` |
| Volume Oscillator | `volume_oscillator.py` | `volume_oscillator(volumes, fast=5, slow=20)` |
| VROC | `vroc.py` | `vroc(volumes, period=14)` |
| Twiggs Money Flow | `twiggs.py` | `twiggs_money_flow(highs, lows, closes, volumes, period=21)` |
| WAD | `wad.py` | `wad(highs, lows, closes)` |
| Volume Profile | `volume_profile.py` | `volume_profile(closes, volumes, bins=20)` / `poc(closes, volumes, period=20)` |
| Volume RSI | `volume_weighted_rsi.py` | `volume_rsi(closes, volumes, period=14)` |
| TVI | `trade_volume_index.py` | `tvi(closes, volumes, min_tick=0.5)` |
| Volume Momentum | `volume_momentum.py` | `volume_momentum(volumes, period=14)` / `relative_volume(volumes, period=20)` |
| Money Flow | `money_flow.py` | `money_flow(h, l, c, v)` / `money_flow_ratio(h, l, c, v, period=14)` |
| Demand Index | `demand_index.py` | `demand_index(highs, lows, closes, volumes, period=14)` |
| Volume Weighted MACD | `volume_weighted_macd.py` | `vwmacd(closes, volumes, fast=12, slow=26, signal=9)` |
| A/D Oscillator | `accumulation_distribution_oscillator.py` | `ad_oscillator(highs, lows, closes, volumes, fast=3, slow=10)` |
| EMV Signal | `ease_of_movement_signal.py` | `emv_signal(highs, lows, volumes, period=14, signal_period=9)` |
| Volume Force | `volume_force.py` | `volume_force(closes, volumes, period=13)` |
| OI Rate of Change | `oi_rate_of_change.py` | `oi_roc(oi, period=14)` |
| PVI Signal | `positive_volume_index_signal.py` | `pvi_signal(closes, volumes, period=255)` |
| Chaikin Oscillator | `chaikin_oscillator.py` | `chaikin_oscillator(highs, lows, closes, volumes, fast=3, slow=10)` |
| Normalized Volume | `normalized_volume.py` | `normalized_volume(volumes, period=20)` |
| Buying/Selling Pressure | `buying_selling_pressure.py` | `buying_selling_pressure(highs, lows, closes, volumes, period=14)` |
| Volume Efficiency | `volume_efficiency.py` | `volume_efficiency(closes, volumes, period=20)` |
| OI Adjusted Volume | `oi_adjusted_volume.py` | `oi_adjusted_volume(volumes, oi, period=20)` |
| OI Flow | `oi_flow.py` | `oi_flow(closes, oi, volumes, period=14)` |
| OI Accumulation | `oi_accumulation.py` | `oi_accumulation(closes, oi, period=20)` |
| OI Climax | `oi_climax.py` | `oi_climax(oi, volumes, period=20, threshold=2.0)` |

## Microstructure — 市场微观结构（15个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Amihud Illiquidity | `amihud.py` | `amihud_illiquidity(closes, volumes, period=20)` |
| Roll Spread | `roll_spread.py` | `roll_spread_estimate(closes, period=20)` |
| Price Impact (Kyle's Lambda) | `price_impact.py` | `kyle_lambda(closes, volumes, period=60)` |
| Volume Clock | `volume_clock.py` | `volume_clock(volumes, target_volume=None, period=20)` |
| Overnight Return | `overnight_return.py` | `overnight_return(opens, closes, period=20)` |
| Trade Intensity | `trade_intensity.py` | `trade_intensity(volumes, period=20)` |
| Price Efficiency | `price_efficiency.py` | `price_efficiency_coefficient(closes, period=20)` |
| Realized Spread | `realized_spread_proxy.py` | `realized_spread(closes, period=20)` |
| Volume Imbalance | `volume_imbalance.py` | `volume_imbalance(closes, volumes, period=20)` |
| High-Low Spread | `high_low_spread.py` | `hl_spread(highs, lows, closes, period=20)` |
| Tick Direction | `tick_direction.py` | `tick_direction(closes, period=20)` |
| Volume Concentration | `volume_concentration.py` | `volume_concentration(volumes, period=20, top_pct=0.2)` |
| Range to Volume | `range_to_volume.py` | `range_to_volume(highs, lows, volumes, period=20)` |
| Trade Clustering | `trade_clustering.py` | `trade_clustering(volumes, period=20)` |
| Adverse Selection | `adverse_selection.py` | `adverse_selection(closes, volumes, period=20)` |

## ML — 机器学习/统计学习（65个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Rolling PCA | `pca_features.py` | `rolling_pca(features_matrix, period=60, n_components=3)` |
| K-Means Regime | `kmeans_regime.py` | `kmeans_regime(features_matrix, period=120, n_clusters=3)` |
| Isolation Forest Anomaly | `isolation_anomaly.py` | `isolation_anomaly(features_matrix, period=120, contamination=0.05)` |
| Gaussian Mixture Regime | `gaussian_mixture_regime.py` | `gmm_regime(features_matrix, period=120, n_components=3)` |
| Transfer Entropy | `transfer_entropy.py` | `transfer_entropy(source, target, period=60, n_bins=5, lag=1)` |
| Ridge Forecast | `ridge_forecast.py` | `rolling_ridge(closes, features_matrix, period=120, forecast_horizon=5)` |
| Lasso Importance | `lasso_importance.py` | `rolling_lasso_importance(closes, features_matrix, period=120)` |
| Granger Causality | `granger_proxy.py` | `granger_causality_score(series_a, series_b, period=60, max_lag=5)` |
| Kalman Trend | `kalman_trend.py` | `kalman_filter(closes, process_noise=0.01, measurement_noise=1.0)` |
| Recurrence Rate | `recurrence.py` | `recurrence_rate(data, period=60, threshold_pct=10)` |
| Adaptive Kalman | `kalman_adaptive.py` | `adaptive_kalman(closes, period=60)` |
| Copula Tail Dependence | `copula_tail.py` | `tail_dependence(returns_a, returns_b, period=120, quantile=0.05)` |
| Hilbert Transform | `hilbert_features.py` | `hilbert_transform_features(closes, period=60)` |
| Bayesian Online Trend | `bayesian_trend.py` | `bayesian_online_trend(closes, hazard_rate=0.01)` |
| Online SGD | `online_regression.py` | `online_sgd_signal(closes, features_matrix, learning_rate=0.01, period=20)` |
| LMS Filter | `adaptive_lms.py` | `lms_filter(closes, reference=None, period=20, mu=0.01)` |
| Autoencoder Error | `autoencoder_error.py` | `reconstruction_error(features_matrix, period=120, encoding_dim=2)` |
| RLS Filter | `rls_filter.py` | `rls_filter(closes, order=5, forgetting=0.99)` |
| KDE Support/Resistance | `kernel_density_levels.py` | `kde_support_resistance(closes, period=60, n_levels=3)` |
| Manifold Embedding | `manifold_features.py` | `manifold_embedding(features_matrix, period=120, n_components=2)` |
| Elastic Net Forecast | `elastic_net_forecast.py` | `elastic_net_signal(closes, features_matrix, period=120, alpha=0.1, l1_ratio=0.5)` |
| Rolling Eigen Features | `rolling_correlation_matrix.py` | `rolling_eigen_features(features_matrix, period=60)` |
| Quantile Regression Bands | `quantile_bands.py` | `quantile_regression_bands(closes, period=60, quantiles=(0.05, 0.5, 0.95))` |
| Regime Persistence | `regime_persistence.py` | `regime_duration(regime_labels, period=60)` |
| Mutual Information | `mutual_information.py` | `rolling_mutual_info(series_a, series_b, period=60, n_bins=10)` |
| Online Covariance | `online_covariance.py` | `exponential_covariance(returns_matrix, halflife=20)` |
| Feature Importance | `feature_importance.py` | `rolling_tree_importance(closes, features_matrix, period=120, n_estimators=50)` |
| Information Ratio | `information_ratio.py` | `rolling_information_ratio(returns, benchmark_returns, period=60)` |
| Spectral Clustering Regime | `spectral_clustering_regime.py` | `spectral_regime(features_matrix, period=120, n_clusters=3)` |
| Shrinkage Covariance | `shrinkage_covariance.py` | `ledoit_wolf_features(returns_matrix, period=120)` |
| Ensemble Signal | `ensemble_signal.py` | `ensemble_vote(closes, features_matrix, period=120)` |
| HMM Regime | `hmm_regime.py` | `hmm_regime(closes, n_states=3, period=252)` |
| Gradient Trend | `gradient_trend.py` | `gradient_signal(closes, period=20, smoothing=5)` |
| Attention Score | `attention_score.py` | `attention_weights(features_matrix, target, period=60)` |
| Momentum Decompose | `momentum_decompose.py` | `momentum_components(closes, short=5, medium=20, long=60)` |
| Robust Z-Score | `robust_zscore.py` | `robust_zscore(data, period=60)` |
| Wavelet Decompose | `wavelet_decompose.py` | `wavelet_features(closes, wavelet="db4", level=4)` |
| Trend Filter | `trend_filter.py` | `trend_filter(closes, period=60, lambda_param=1.0)` |
| Volatility Forecast | `volatility_forecast.py` | `garch_like_forecast(closes, period=60, alpha=0.1, beta=0.85)` |
| Cross Validation Signal | `cross_validation_signal.py` | `cv_signal_strength(closes, features_matrix, period=120, n_folds=3)` |
| Decision Boundary | `decision_boundary.py` | `decision_boundary_distance(features_matrix, labels, period=120)` |
| KNN Signal | `nearest_neighbor_signal.py` | `knn_signal(closes, features_matrix, period=120, k=5)` |
| Incremental PCA | `incremental_pca.py` | `incremental_pca_signal(features_matrix, n_components=3)` |
| Random Projection | `random_projection.py` | `random_projection_features(features_matrix, n_components=5, period=120)` |
| Target Encoding | `target_encoding.py` | `target_encoded_regime(closes, period=60, n_bins=5)` |
| Correlation Network | `rolling_correlation_network.py` | `correlation_network_score(returns_matrix, period=60, threshold=0.5)` |
| Online Mean/Variance | `online_mean_variance.py` | `welford_stats(data)` |
| CUSUM Filter | `cusum_filter.py` | `cusum_event_filter(closes, threshold=1.0)` |
| Multi Entropy | `entropy_features.py` | `multi_entropy(closes, period=60)` |
| DTW Distance | `dynamic_time_warping.py` | `dtw_distance(series_a, series_b, period=60)` |
| Prophet-Like Trend | `prophet_like_trend.py` | `piecewise_trend(closes, n_changepoints=5, period=252)` |
| Boosting Signal | `boosting_signal.py` | `gradient_boost_signal(closes, features_matrix, period=120, n_estimators=20)` |
| Variational Inference | `variational_inference.py` | `variational_regime(closes, period=120, n_components=3)` |
| Rolling Percentile Rank | `rolling_percentile_rank.py` | `percentile_rank_features(features_matrix, period=60)` |
| Symbolic Regression | `symbolic_regression_signal.py` | `symbolic_features(closes, period=60)` |
| Model Disagreement | `disagreement_index.py` | `model_disagreement(closes, features_matrix, period=120)` |
| Regime Transition Matrix | `regime_transition_matrix.py` | `transition_features(regime_labels, n_states=3)` |
| Distance Correlation | `nonlinear_correlation.py` | `distance_correlation(series_a, series_b, period=60)` |
| Von Neumann Ratio | `successive_differences.py` | `von_neumann_ratio(data, period=60)` |
| Fractal Market Hypothesis | `fractal_market.py` | `fractal_market_hypothesis(closes, period=120)` |
| OI PCA | `oi_pca.py` | `oi_pca_features(closes, oi, volumes, period=120)` |
| OI Anomaly | `oi_anomaly.py` | `oi_anomaly(closes, oi, volumes, period=120)` |
| OI Prediction | `oi_prediction.py` | `oi_predicted_return(closes, oi, volumes, period=120)` |
| OI Cluster | `oi_cluster.py` | `oi_cluster(closes, oi, volumes, period=120, n_clusters=4)` |
| OI Kalman | `oi_kalman.py` | `oi_kalman_trend(oi, process_noise=0.01, measurement_noise=1.0)` |

## Regime — 行情状态识别（33个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Structural Break (CUSUM) | `structural_break.py` | `cusum_break(data, period=60, threshold=3.0)` |
| Changepoint Score | `changepoint.py` | `changepoint_score(data, period=60)` |
| Distribution Shift (KL) | `distribution_shift.py` | `kl_divergence_shift(data, period=60, reference_period=120)` |
| Tail Index (Hill) | `tail_index.py` | `hill_tail_index(data, period=60, k_fraction=0.1)` |
| Mean Reversion Speed (OU) | `mean_reversion_speed.py` | `ou_speed(data, period=60)` |
| Trend Persistence | `trend_persistence.py` | `trend_persistence(data, max_lag=20, period=60)` |
| Volatility Clustering | `volatility_clustering.py` | `vol_clustering(data, period=60)` |
| Variance Ratio | `variance_ratio.py` | `variance_ratio_test(data, period=60, holding=5)` |
| Sample Entropy | `sample_entropy.py` | `sample_entropy(data, m=2, r_mult=0.2, period=60)` |
| Fractal Dimension | `fractal_dimension.py` | `fractal_dim(data, period=60)` |
| Spectral Density | `spectral_density.py` | `dominant_cycle(data, period=120)` |
| Turbulence Index | `turbulence_index.py` | `turbulence(returns_matrix, period=60)` |
| Composite Regime Score | `regime_score.py` | `composite_regime(closes, highs, lows, period=20)` |
| Market State | `market_state.py` | `market_state(closes, volumes, oi, period=20)` |
| Jump Detector | `jump_detector.py` | `jump_detection(closes, period=20, threshold=3.0)` |
| Trend Strength Composite | `trend_strength_composite.py` | `trend_strength(closes, highs, lows, period=20)` |
| Mean-Variance Regime | `mean_variance_regime.py` | `mv_regime(data, period=60, n_regimes=3)` |
| Momentum Regime | `momentum_regime.py` | `momentum_regime(closes, fast=10, slow=60)` |
| Correlation Breakdown | `correlation_breakdown.py` | `correlation_breakdown(returns_a, returns_b, period=60, stress_threshold=2.0)` |
| Efficiency Ratio | `efficiency_ratio.py` | `efficiency_ratio(closes, period=20)` |
| Complexity Profile | `complexity_profile.py` | `complexity_profile(closes, scales=None)` |
| Mean Crossing Rate | `mean_crossing_rate.py` | `mean_crossing(closes, period=60)` |
| Runs Test | `runs_test.py` | `runs_test(closes, period=60)` |
| Hurst R/S | `hurst_rs.py` | `hurst_rs(data, min_period=10, max_period=100)` |
| Adaptive Lookback | `adaptive_period.py` | `adaptive_lookback(closes, min_period=10, max_period=100)` |
| Stationarity Score | `stationarity_score.py` | `stationarity(closes, period=60)` |
| Vol Regime Markov | `vol_regime_markov.py` | `vol_regime_simple(closes, period=60)` |
| Entropy Rate | `entropy_rate.py` | `entropy_rate(closes, period=60, m=3)` |
| Price Inertia | `price_inertia.py` | `price_inertia(closes, period=20)` |
| Regime Switch Speed | `regime_switch_speed.py` | `switch_speed(regime_labels, period=60)` |
| OI Regime | `oi_regime.py` | `oi_regime(closes, oi, volumes, period=60)` |
| OI Cycle | `oi_cycle.py` | `oi_cycle(oi, period=60)` |
| OI Stress | `oi_stress.py` | `oi_stress(closes, oi, volumes, period=20)` |

## Seasonality — 季节性/日历效应（15个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Monthly Seasonal | `monthly_pattern.py` | `monthly_seasonal(closes, datetimes, lookback_years=3)` |
| Weekday Effect | `weekday_effect.py` | `weekday_effect(closes, datetimes, lookback=252)` |
| Quarter-End Effect | `quarter_effect.py` | `quarter_end_effect(closes, datetimes, window=10)` |
| Holiday Proximity | `holiday_proximity.py` | `holiday_effect(closes, datetimes, lookback=252)` |
| Seasonal Decompose | `seasonal_decompose.py` | `seasonal_strength(closes, period=252)` |
| Contract Roll | `contract_roll.py` | `roll_effect(closes, is_rollover, period=20)` |
| Year Progress | `year_progress.py` | `year_cycle(datetimes)` |
| Month of Year | `month_of_year.py` | `month_cycle(datetimes)` |
| Trading Day Number | `trading_day_number.py` | `trading_day_of_month(datetimes)` |
| Seasonal Momentum | `seasonal_momentum.py` | `seasonal_momentum(closes, datetimes, lookback_years=3)` |
| Intraweek Pattern | `intraweek_pattern.py` | `intraweek_momentum(closes, datetimes, lookback=52)` |
| Month Turn Effect | `month_turn_effect.py` | `month_turn(closes, datetimes, window=3)` |
| Seasonal Z-Score | `seasonal_zscore.py` | `seasonal_zscore(closes, datetimes, period=252)` |
| Volatility Seasonality | `volatility_seasonality.py` | `vol_seasonality(closes, datetimes, vol_period=20)` |
| Expiry Week | `expiry_week.py` | `expiry_week_effect(closes, datetimes, lookback=52)` |

## Spread — 跨品种价差/比值（25个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Gold/Silver Ratio | `gold_silver_ratio.py` | `gold_silver_ratio(au_closes, ag_closes, period=60)` |
| Metal Ratio | `metal_ratio.py` | `metal_ratio(closes_a, closes_b, period=60)` |
| Basis | `basis.py` | `basis(front_closes, back_closes, period=20)` |
| Cross Momentum | `cross_momentum.py` | `cross_momentum(closes_a, closes_b, period=20)` |
| Pair Z-Score | `pair_zscore.py` | `pair_zscore(closes_a, closes_b, period=60, method="ratio")` |
| Cointegration Residual | `cointegration_residual.py` | `cointegration_residual(closes_a, closes_b, period=120)` |
| Relative Strength | `relative_strength.py` | `relative_strength(asset_closes, benchmark_closes, period=20)` |
| Correlation Regime | `correlation_regime.py` | `correlation_regime(closes_a, closes_b, fast=20, slow=60)` |
| Lead-Lag | `lead_lag.py` | `lead_lag(closes_a, closes_b, max_lag=10, period=60)` |
| Rolling Beta | `beta.py` | `rolling_beta(asset_returns, benchmark_returns, period=60)` |
| Sector Momentum | `sector_momentum.py` | `sector_momentum(closes_list, period=20)` |
| Term Premium | `term_premium.py` | `term_premium(front_closes, back_closes, period=20)` |
| Hedging Pressure | `hedging_pressure.py` | `hedging_pressure(closes, oi, volumes, period=20)` |
| Contagion Score | `contagion.py` | `contagion_score(returns_a, returns_b, period=60, threshold=2.0)` |
| Dispersion | `dispersion.py` | `dispersion(closes_list, period=20)` |
| Energy/Metal Ratio | `energy_metal_ratio.py` | `energy_metal_ratio(energy_closes, metal_closes, period=60)` |
| Intermarket Divergence | `intermarket_divergence.py` | `intermarket_divergence(closes_a, closes_b, period=20)` |
| Ratio Momentum | `ratio_momentum.py` | `ratio_momentum(closes_a, closes_b, period=20, lookback=60)` |
| Spread Volatility | `spread_volatility.py` | `spread_volatility(closes_a, closes_b, period=20)` |
| Cross Asset RSI | `cross_asset_rsi.py` | `cross_asset_rsi(closes_a, closes_b, period=14)` |
| Common Factor | `common_factor.py` | `common_factor(closes_list, period=60)` |
| Residual Momentum | `residual_momentum.py` | `residual_momentum(asset_closes, factor_closes, period=60, mom_period=20)` |
| Dynamic Hedge Ratio | `dynamic_hedge_ratio.py` | `dynamic_hedge(closes_a, closes_b, period=60)` |
| Carry Signal | `carry.py` | `carry_signal(front_closes, back_closes, period=20)` |
| Relative Value Z-Score | `relative_value_zscore.py` | `rv_zscore(closes_a, closes_b, closes_c, period=60)` |

## Structure — 持仓结构分析（23个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| OI-Price Regime | `oi_price_regime.py` | `oi_price_regime(closes, oi, period=20)` |
| OI Concentration | `oi_concentration.py` | `oi_concentration(oi, period=60)` |
| Volume/OI Ratio | `volume_oi_ratio.py` | `volume_oi_ratio(volumes, oi, period=20)` |
| Smart Money Index | `smart_money.py` | `smart_money_index(opens, closes, highs, lows, volumes, period=20)` |
| Delivery Pressure | `delivery_pressure.py` | `delivery_pressure(oi, volumes, datetimes, period=20)` |
| Position Crowding | `position_crowding.py` | `position_crowding(closes, oi, volumes, period=60)` |
| Net Positioning | `net_positioning.py` | `net_positioning_proxy(closes, oi, volumes, period=20)` |
| Squeeze Detector | `squeeze_detector.py` | `squeeze_probability(closes, oi, volumes, period=20)` |
| Warehouse Proxy | `warehouse_proxy.py` | `inventory_proxy(closes, oi, volumes, period=40)` |
| Speculation Index | `speculation_index.py` | `speculation_index(volumes, oi, period=20)` |
| OI Velocity | `oi_velocity.py` | `oi_velocity(oi, period=5)` |
| PVT Strength | `price_volume_trend_strength.py` | `pvt_strength(closes, volumes, period=20)` |
| Market Depth Proxy | `market_depth_proxy.py` | `depth_proxy(highs, lows, volumes, period=20)` |
| Commitment Ratio | `commitment_ratio.py` | `commitment_ratio(oi, volumes, period=20)` |
| OI Divergence Enhanced | `oi_divergence_enhanced.py` | `oi_divergence_enhanced(closes, oi, volumes, period=20)` |
| OI Bollinger | `oi_bollinger.py` | `oi_bollinger(oi, period=20, num_std=2.0)` |
| OI Relative Strength | `oi_relative_strength.py` | `oi_relative_strength(oi, volumes, period=20)` |
| OI Mean Reversion | `oi_mean_reversion.py` | `oi_mean_reversion(oi, period=60)` |
| OI Breakout | `oi_breakout.py` | `oi_breakout(oi, period=20, threshold=2.0)` |
| OI-Volume Divergence | `oi_volume_divergence.py` | `oi_volume_divergence(oi, volumes, period=20)` |
| OI Momentum-Price Divergence | `oi_momentum_divergence.py` | `oi_momentum_price_divergence(closes, oi, period=20)` |
| OI Weighted Price | `oi_weighted_price.py` | `oi_weighted_price(closes, oi, period=20)` |
| OI Persistence | `oi_persistence.py` | `oi_persistence(oi, period=20)` |

---

## 汇总

| 分类 | 数量 | 说明 |
|------|:---:|------|
| momentum | 35 | 动量/振荡类 |
| trend | 35 | 趋势类 |
| volatility | 35 | 波动率类 |
| volume | 39 | 成交量/持仓类 |
| microstructure | 15 | 市场微观结构 |
| ml | 65 | 机器学习/统计学习 |
| regime | 33 | 行情状态识别 |
| seasonality | 15 | 季节性/日历效应 |
| spread | 25 | 跨品种价差/比值 |
| structure | 23 | 持仓结构分析 |
| **总计** | **320** | |
