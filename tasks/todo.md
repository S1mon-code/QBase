# QBase TODO

## Completed
- [x] Step 1: 项目骨架
- [x] Step 2: 100 个核心指标 (momentum 25 + trend 25 + volatility 25 + volume 25)
- [x] Step 3: 指标单元测试 (基础覆盖: momentum, trend, volatility, volume)
- [x] Step 4: 50 个强趋势策略 + Optuna 优化 + 测试集验证 (49/50 正 Sharpe)
- [x] Step 4b: 预计算架构迁移 (on_init_arrays, ~20x 加速)
- [x] Step 4.5: 归因分析模块 (attribution/)
  - 信号归因: ablation test，逐指标测试贡献度
  - 行情归因: ADX/ATR/Volume 三维 regime 标注 + 分 regime 统计
  - 自动集成到 validate_and_iterate.py
  - v12 首个归因报告: Volume Momentum 贡献 51.7%，Aroon/PPO 近乎冗余
- [x] Step 5: Portfolio 构建框架
  - 通用 builder: `portfolio/builder.py` (穷举/贪心 + Ledoit-Wolf + Sharpe 加权 HRP + 活跃度过滤 + Bootstrap)
  - 通用 scorer: `portfolio/scorer.py` (4维16指标评分)
  - Strong Trend Portfolio C (通用): AG Sharpe 2.58, LC Sharpe 2.37
- [x] Step 8: 全时间策略 (all_time/ag/)
  - 100 个多空策略 (5类: 趋势/均值回归/突破/多周期/混合)
  - 自动参数检测 optimizer
  - 训练集 97/100 正 Sharpe, 测试集 51/100 正 Sharpe

## Next
- [ ] Step 4.5b: 对 Portfolio C 全部策略跑归因 (v8, v11, v31, v34)
- [ ] Step 5b: Walk-forward 验证 — 5年滚动训练→1年测试，确认权重稳定性
- [ ] Step 6: 品种筛选器 (screener/) — 趋势/震荡/突破模式扫描
- [ ] Step 7: 中趋势策略 (medium_trend/) — 涨幅 20-80%
- [ ] Step 8b: 全时间策略 — 更多品种 (all_time/i/, all_time/cu/ 等)
- [ ] Step 9: 模拟盘对接
