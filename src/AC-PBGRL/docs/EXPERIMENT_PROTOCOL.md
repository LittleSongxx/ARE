# AC-PBGRL 公平实验与论文素材协议

## 1. 预注册比较

- 二维学习轨：`ariadne`、`ariadne_pi`、`q_distillation`、`full`，每项 5 个训练种子。
- 机制消融：MSE、NLL、NLL+Rank、KF、diffusion/context、EMA、GRU，每项 3 个训练种子。
- ROS/Gazebo 轨：固定版本 TARE、导出 ARiADNE、导出 AC-PBGRL；不把二维 simulator reward 与真实时间指标混为一表。
- 所有学习方法使用相同地图 split、sensor range、可执行邻居动作、全局 batch 128、1,000,000 transition 预算与 0.0625 update/transition。
- replay warm-up transition 计入统一环境交互预算，但不产生梯度更新，也不进入后续 update-credit 计算；所有方法使用完全相同的 warm-up 门槛。
- 训练 GPU 数只影响墙钟时间。规划延迟在同一固定单卡、相同进程优先级和 warm-up 后重新测量。

## 2. 数据防泄漏

`splits` 对每张地图计算旋转/翻转 D4 canonical hash；等价地图只能属于同一 split。复杂度最高的预注册 10% group 作为 OOD，其余按复杂度分层分配 train/validation/IID test。manifest 保存原图 SHA256、split hash、几何复杂度和分组。

GT map 只用于：

1. privileged Critic 的训练输入；
2. 冻结 teacher 的 future-gain 标签；
3. 评测覆盖率真值。

Actor、KF、EMA、GRU 和 ONNX 部署端只读取 belief graph。标签定义为六步、`gamma=0.95` 的 baseline frontier-minus-distance reward，不含终止 bonus；同一 rollout 中 simulator belief 更新自然去除重复传感器覆盖。

## 3. 主要任务指标

| 字段 | 定义 | 方向 |
|---|---|---|
| `episode/success` | 达到完成条件的比例 | 高 |
| `episode/completion_distance` | 成功 episode 的总行驶距离 | 低 |
| `episode/coverage_95_distance` | 首次达到 95% coverage 的距离 | 低 |
| `episode/explored_rate` | 截止时最终覆盖率 | 高 |
| `episode/makespan_steps` | 决策/waypoint 步数 | 低 |
| `episode/backtracks` | 立即返回前一节点次数 | 低 |
| `episode/repeated_edges` | 重复有向边次数 | 低 |
| `episode/direction_switches` | 相邻运动向量夹角大于 90° 次数 | 低 |
| `system/planning_latency_*` | 单次两阶段 potential/KF/Pointer 推理耗时 | 低 |
| `graph/nodes_*` | 实际编码节点数 | 诊断 |
| `system/peak_gpu_memory_mib` | 固定评测设备峰值显存 | 低 |

不能把失败 episode 的短路程当成效率优势。论文同时报告成功率；完成距离只对成功样本解释，95% coverage 未达到时记缺失而不是记 0。

## 4. Potential 与滤波诊断

- RMSE、MAE、Gaussian NLL；
- 50/80/90/95% interval coverage 和方差分箱 reliability；
- Spearman、Kendall、pairwise accuracy、top-1 regret；
- KF innovation/NIS、事件数、reset/retire 数、active record 数；
- 按 IID/OOD、地图复杂度、候选数量和 episode 阶段分层。

Action variance 是 region variance 与非负 residual variance 之和。validation temperature 满足 `T_region <= T_action`，从而在替换 region posterior 后仍能组成非负 action uncertainty。此方差是条件预测不确定性，不直接声称 epistemic/OOD uncertainty。

## 5. 统计协议

- 配对 key 固定为 `(seed, map_id, split)`；
- 主结果报告配对 bootstrap 95% CI；
- 双侧 Wilcoxon signed-rank；
- 同一结果族采用 Holm 校正；
- 报告 Cliff's delta 作为效应量；
- 不根据测试集结果调整 horizon、标签定义、地图 split 或完成阈值。

`figures` 自动输出学习曲线、主指标点图、配对 forest plot、校准曲线、ranking 图、path behavior、latency—graph scale、代表性同图路径和 distance—latency Pareto 图，同时保存用于复核的 CSV。

## 6. 必做消融顺序

1. ARiADNE belief Critic；
2. ARiADNE+PI；
3. Q distillation；
4. scalar potential MSE；
5. heteroscedastic NLL；
6. NLL + RankNet；
7. calibrated KF，并与 no-memory/EMA/GRU 对照；
8. multiscale diffusion/context；
9. 完整方法。

只有逐项带来预测、行为或最终任务收益时，才在论文中保留对应贡献。A*、普通图稀疏化、GT Critic 和组件名称本身不作为新颖性主张。
