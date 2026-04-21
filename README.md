# DACI Simulator — V1

第一版 Python 模拟器，实现论文 `DACI: Drift-Aware Collaborative LLM Inference` 的数值实验框架。所有参数都集中在 `configs/*.json`，便于调参和复现。

## 快速开始

```bash
cd daci_sim
python run.py --n_traces 2 --schemes DACI,SDA --verbose
```

输出保存在 `outputs/run_<timestamp>/`：
- `config_snapshot.json` — 本次运行使用的**完整**配置
- `summary.csv` — 每个 scheme 的聚合指标
- `traces/<scheme>_seed<N>.jsonl` — 每 window 的完整状态 dump

## 目录

```
daci_sim/
├── configs/              # 全部可调参数（JSON）
│   ├── devices.json      # 设备 tier、FLOPS、热/工作负载敏感度
│   ├── models.json       # LLM 配置 + FLOPs 公式常数 + effective_utilization
│   ├── drift.json        # GT 漂移参数 + regime 切换（R1/R2/R3/R4）
│   ├── algo.json         # DACI 超参：W, H_max, τ, S_range, λ, latency_model
│   └── experiment.json   # traces 数、seed、schemes、请求 P/G
├── src/
│   ├── config.py         # JSON 加载 + 快照
│   ├── model_spec.py     # 每块 FLOPs/weight/KV
│   ├── cluster.py        # 节点与链路基线
│   ├── cost_model.py     # Eqs.(3)-(8)：C_s, D_s, H_s, Ω_b, T_startup
│   ├── drift_gt.py       # 地面真值：M/G/∞ workload, 非对称热, Markov 网络
│   ├── predictor.py      # Kalman + AR(1) + adaptive horizon
│   ├── dp.py             # Bottleneck-augmented DP（命题 1）+ init/runtime 实例
│   ├── schemes/schemes.py# DACI / SDA / RT / FM / OR
│   ├── simulator.py      # 单 trace 执行循环
│   └── metrics.py        # TTLT / TTFT / P99 TPOT / Ovhd 聚合 + JSONL 日志
└── run.py                # CLI 入口
```

## 已实现的关键对应

| 论文内容 | 文件 / 函数 |
|---|---|
| Eq.(3) 计算时延 C_s | `cost_model.C_stage` |
| Eq.(4) 热漂移 φ_th | `drift_gt.compute_true_phi` + `predictor.forecast` |
| Eq.(5) 工作负载漂移 φ_wk | 同上 |
| Eq.(6) Hockney 通信 D_s | `cost_model.D_stage` |
| Eq.(7)-(8) 切换开销 Ω | `cost_model.H_stage` / `Omega_reconfig` |
| Eq.(11) 热 RC 动态 | `predictor.Predictor.kalman_update` / `forecast` |
| Eq.(12) AR(1) 工作负载 | `predictor.rls_update` / `forecast` |
| Eq.(15) 自适应 H_r* | `predictor.adaptive_horizon` |
| 命题 1 Bottleneck DP 模板 | `dp.solve_initial_dp` / `solve_runtime_dp` |
| Eq.(17)-(18) 初始部署 DP | `dp.solve_initial_dp` |
| Eq.(19)-(20) 运行时边界 DP | `dp.solve_runtime_dp` |
| Eq.(22) 惰性切换 | `schemes.DACIScheme.decide_runtime` 中的 J_new < J_inc 比较 |
| Algorithm 1 总流程 | `simulator.run_trace` |

## 已确认的建模决策

1. **Latency model**：`sum`（论文 Eq.(6) 字面值），在 `algo.json` 留开关。
2. **权重驻留**：`strict` — 节点只驻留当前 b 下的块；边界扩张时对新块走 L_n(Δ) 成本（与 Eq.(7) 一致）。
3. **per-block 粒度**：均匀拆分（ω_model/L）；embedding 在 N_0、LM head 在 a_S 都不计入 per-block。
4. **末端 → N_0 回传**：忽略。
5. **N_0**：不建模。
6. **W_sec**：`W · TPOT_baseline`，TPOT_baseline 由上一窗口 DP 在 φ=1 下算出。
7. **热 GT 非对称 τ_up/τ_down**：实现；Kalman 用 τ_bar（调和均值），故意 mis-specified。
8. **workload GT 用 M/G/∞**，predictor 用 AR(1)，故意 mis-specified。
9. **初始温度**：θ = θ_amb + ν·q + 𝒩(0, σ_v²)。
10. **Markov self-loop**：normal→normal=0.998，degraded→degraded=0.95。
11. **外生随机源 (到达/时长/网络/sensor)** 按 seed 固定跨 scheme 共享；u 由 scheme 注入进热演化。
12. **κ^dec per block = 2d² + 4d(P+t)**；κ^pf = 2Pd² + 2P²d。
13. **effective_utilization = 0.15 (decode) / 0.35 (prefill)**，可在 `models.json` 覆盖。

## 当前已知的 caveats（V1 预期）

1. **初始 DP 耗时** ~30s/trace。主要是 `P(8, S)` 排列枚举 × 每排列小 DP。如果想更快可以缩小 `S_max` 或提前剪枝。
2. **首轮冒烟结果 TPOT 太低**（~1ms/token）。原因是默认 `eff_util_decode=0.15` 对 decode 仍偏乐观——实测边缘 LLM decode 常 memory-bound，实际利用率可能 0.01–0.05。你应在 `models.json` 中按你的 prototype 实测调整。
3. **默认 regime 下 DACI 不切换**：热上升不到阈值。跑 R2 (thermal-dominant) 应能看到切换行为。把 `drift.json` 的 `regime.active` 改成 `"R2_thermal"` 并把 G_hat 调大到 4096 试试。
4. **RT 和 FM baseline** 已实现但未完整测试；建议先验证 DACI/SDA 行为后再启用。
5. **OR (Oracle)** 的 future phi 是用"无 self-heating"的 GT 预先算的，是近似而非真实最优——充分但不紧的下界。
6. **Memory constraint** 在 Gemma-7B 默认配置下不会触发；要触发需要更大模型（llama-13b 或 qwen-32b）。

## 调试建议

先跑：
```bash
python run.py --n_traces 2 --schemes DACI,SDA
```
预期：TTLT 相同（默认 regime drift 太小，DACI 无切换）。

再把 `configs/drift.json` 的 `regime.active` 改为 `"R2_thermal"`，
把 `configs/experiment.json` 的 `G_hat_tokens` 改为 `4096`，再跑：
```bash
python run.py --n_traces 2 --schemes DACI,SDA --verbose
```
预期：DACI TTLT < SDA，#Reconf > 0。

## 下一步（V2 候选）

- [ ] RT / FM / OR 完整验证
- [ ] `latency_model: "max"` 和 `"mixed"` 实现
- [ ] 完整的 regime sweep (R1-R4) + 可视化脚本
- [ ] Predictor RMSE 评估（Sec. 5.5）
- [ ] 可选：带 KV-cache compression 的 FM 变体（Conclusion 提到）
