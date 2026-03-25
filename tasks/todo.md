# QBase TODO

## Completed
- [x] Step 1: 项目骨架
- [x] Step 2: 100 个核心指标 (momentum 25 + trend 25 + volatility 25 + volume 25)
- [x] Step 3: 指标单元测试 (基础覆盖: momentum, trend, volatility, volume)
- [x] Step 4: 50 个强趋势策略 + Optuna 优化 + 测试集验证 (49/50 正 Sharpe)
- [x] Step 4b: 预计算架构迁移 (on_init_arrays, ~20x 加速)

## Next
- [ ] Step 5: Portfolio 构建 — 筛选 + 相关性过滤 + HRP 赋权 + Walk-forward
- [ ] Step 6: 品种筛选器 (screener/) — 趋势/震荡/突破模式扫描
- [ ] Step 7: 中趋势策略 (medium_trend/) — 涨幅 20-80%
- [ ] Step 8: 全时间策略 (all_time/ag/) — AG 全天候多空
- [ ] Step 9: 模拟盘对接
