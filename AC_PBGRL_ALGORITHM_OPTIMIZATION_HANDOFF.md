# AC-PBGRL 算法诊断与优化 Handoff

最后更新：2026-08-28 16:29（Asia/Shanghai）
本地分支：`dev_kf`
编写前本地 HEAD：`245a0d64`
最近的实现变更：`90705256`（decoded future-gain label shard cache；只优化吞吐，不改变算法语义）

> **当前是硬暂停状态。不要自动恢复训练、评测或 cron。**
> 用户因 `full@200k` 的 5 图行为预览明显劣于 `ariadne_pi@200k`，明确要求停止服务器上的全部 AC-PBGRL 进程。新窗口的任务应先定位负迁移根因、修正算法与评测可观测性，得到用户确认后才能重新启动任何服务器实验。

## 1. 一句话结论

`ariadne_pi/seed_0` 与 `full/seed_0` 都已稳定训练到 200k，checkpoint、replay、优化器和全部数值审计均健康；但同图 5-map 行为预览中，`full` 的平均探索率从 baseline 的 `24.63%` 降至 `14.38%`，平均回报从 `-29.69` 降至 `-35.56`，5/5 张地图都没有改善。当前证据更像是**算法/策略融合负迁移或评测语义问题**，不是 NaN、OOM、训练中断或权重没有更新。

不能把“训练数值健康”解释为“优化有效”。也不能用尚未完成的 full 100-map 正式评测声称任何收益。

## 2. 用户当前授权边界

当前允许：

- 阅读和审计本地源码、配置、已有日志和已固化产物；
- 在本地添加针对性单元测试、诊断开关和算法修复；
- 运行本地 CPU/synthetic 测试；
- 形成最小、可证伪的改进方案与重新验证计划；
- 在用户再次明确授权后，执行小规模服务器诊断或重新训练。

当前不允许：

- 恢复 200k→500k 训练；
- 重跑 full 100-map、启动新 seed、1M 正式训练或任何完整消融；
- 恢复服务器 crontab/watchdog；
- 为了“补结果”私自运行额外服务器测试；
- 删除或覆盖现有 checkpoint、replay、labels、pilot、preview 或 archive；
- 使用或结束 GPU 1/2 上的外部任务。

如果新窗口准备调用服务器，必须先向用户说明：运行目的、地图数、是否更新权重、预计资源/时间、输出路径，并取得明确同意。

## 3. 当前服务器状态（停止后的最终快照）

停止时间：2026-08-28 16:25～16:29（Asia/Shanghai）。

- 原 pipeline 进程组 `36186`（watchdog、paper driver、full evaluator）已收到 `SIGTERM` 并退出。
- cron 随即拉起了新进程组 `61057`；确认根因后，该进程组也已收到 `SIGTERM` 并退出。
- 用户 crontab 中完整的 `# BEGIN AC-PBGRL WATCHDOG`～`# END AC-PBGRL WATCHDOG` 块已经移除，因此每分钟和 `@reboot` 都不会再启动 pipeline。
- 原 crontab 临时备份为 `/tmp/acpbgrl-crontab-before-stop.Zj5z1M`。`/tmp` 不是长期存储，不能假设该文件永久存在；即使存在，也不得未经用户授权恢复。
- 16:29 跨过新的 cron 分钟边界后再次检查：没有 `server_watchdog.sh`、`ac_pbgrl.cli paper/train/evaluate/labels` 进程，crontab 为空。
- GPU 0/3 已释放。
- GPU 1/2 当时仍有其他用户进程（停止时 PID 分别为 `9086`、`59919`，各约 12 GiB）；这些 PID 可能变化，任何时候都必须重新核验，且本项目不得触碰。
- 所有已有训练和预览产物均保留；没有删除任何数据。

远端连接地址、登录用户名和密码不得写入本文件或 Git。需要连接时，从用户提供的信息获取。已知远端布局如下，主要用于识别既有产物：

```text
REMOTE_PROJECT_ROOT=/home/user/songensheng/AC-PBGRL
ACPBGRL_DATA_ROOT=/mnt/songensheng/ac-pbgrl
ACPBGRL_PYTHON=/mnt/songensheng/ac-pbgrl/env/bin/python
```

远端源码目录没有 `.git`；本地仓库才是 Git 真源。

## 4. 已完成并应保留的产物

### 4.1 Teacher 与标签

- 共享 `ariadne_pi` teacher：100,000 environment transitions、6,125 optimizer updates、807 episodes。
- train future-gain labels：20,000 states / 20 shards / 325,820 个有效 action targets。
- validation labels：2,048 states / 33,414 个有效 action targets。
- train/validation 与 IID/OOD split 已做零重叠审计；teacher、split provenance 哈希一致。
- train 标签均值约 `-0.832`、标准差约 `0.761`、范围 `[-3.746, 2.913]`，未退化成常数。

### 4.2 `ariadne_pi/seed_0` 200k

```text
checkpoint: $ACPBGRL_DATA_ROOT/runs/ariadne_pi/seed_0/checkpoints/latest.pt
environment_steps: 200000
episodes: 1628
update_step: 12375
update_credit: 0
SHA256: 91ba350b9e5fc6be073adbff0677a3d25b827de865d2b0b7263235d5746a9aa2
```

检查点、内嵌 replay 与磁盘 replay 完全一致；全部浮点张量有限，双 Q 不相同，Actor/Q1/Q2/alpha optimizer step 都是 12,375。

### 4.3 `full/seed_0` 200k

```text
checkpoint: $ACPBGRL_DATA_ROOT/runs/full/seed_0/checkpoints/latest.pt
environment_steps: 200000
episodes: 1617
update_step: 12375
update_credit: 0
temporal: kf
SHA256: 38431cdf6bc59d970089239f791acd8c9e9223faf368119a7350dd9f5465f044
```

审计结果：

- checkpoint、内嵌 replay、磁盘 replay 都精确为 200,000；
- 1,236 个浮点/复数张量、15,435,045 个元素全部有限；
- 双 Q 的 84/84 个可比张量都不同；
- Actor/Q1/Q2/alpha optimizer step 都是 12,375；
- state/next-state 中 151,647/151,378 个有效 KF posterior mean/variance 全部有限，variance 严格为正；
- 1,652 条 metrics、32,361 个数值字段无 NaN/Inf；
- 最终 aggregate：Q1/Q2 loss `9.913/9.793`、Q gradient `3945.10`、target `50.17`、entropy `2.647`、alpha `0.294`；
- auxiliary weight 仅为 `0.04479`（目标 `0.2`），offline potential/region/rank loss 为 `-0.01262/-0.00342/0.60628`。

小幅负 Gaussian NLL 是省略常数项且方差小于 1 时的合法结果，不是本次行为退化的直接证据。

## 5. 现有评测证据

### 5.1 `ariadne_pi@200k` 正式固定 100 图结果

路径：

```text
$ACPBGRL_DATA_ROOT/pilot/step_200000/runs/ariadne_pi/seed_0/evaluation/
```

完整性：100 行、100 张唯一 IID-test 地图、seed 0，所有已写数值有限。

汇总：

```text
success rate:                 0.0000
mean explored rate:           0.256070
mean return:                 -32.605913
mean steps:                 128
mean travel distance:        560.012
mean episode wall time:        8.917 s
mean planning latency:        10.516 ms
episodes.csv SHA256: d9b89501f88b3f65d1be1de7495f4ef270b46c2369a3b88406e3e6083a57ecc4
```

重要异常信号：100/100 都跑满 128 步且成功率为 0。baseline 自身已经落在 success 指标的地板上，因此在优化 full 前，必须确认测试地图难度、128-step 上限、95% completion threshold、训练/评测环境一致性及 baseline 实现是否合理。

### 5.2 `full@200k` 正式 100 图评测没有完成

原进程使用同一批地图和 seed 0，开启 full potential diagnostics。它运行约 4 小时 58 分钟后，按用户指令停止。

evaluator 当前会把全部地图结果留在内存中，直到 100 图结束才统一写 `episodes.csv/jsonl`、`paths.jsonl`、`potential_samples.npz` 和 `summary.json`。因此：

- 停止时 full 正式 evaluation 目录仍为空；
- 没有可恢复的逐图结果；
- 不能虚构完成百分比或从该进程得出 full 的正式 100 图效果；
- 下次正式评测前应先实现逐图原子落盘、进度记录和 resume，避免再次损失数小时计算。

### 5.3 `full@200k` 5 图 behavior-only 预览

为了在停止前快速查看方向，曾用同一 full checkpoint、seed 0 和正式地图列表的前 5 张图做一次**不更新权重**的轻量预览：

- physical GPU 3（通过 `CUDA_VISIBLE_DEVICES=3` 映射为进程内 `cuda:0`）；
- `evaluation.potential_diagnostics=false`；
- `map-limit=5`；
- CPU 数学库线程限制为 1，未干扰主评测的 GPU0；
- 输出不是新的训练 run，也不是论文级重复实验；
- CuBLAS 打印了“未保证逐比特确定性”的 warning，因此它只能作为方向性样例，正式结论仍需统一设备与完整协议。

产物：

```text
$ACPBGRL_DATA_ROOT/previews/step_200000/full_seed_0_first5_behavior_only/
  episodes.csv
  episodes.jsonl
  paths.jsonl
  summary.json
  paired_preview.png

episodes.csv SHA256: 7c53c625bb2f8ca3e83435fe30b749b8a409f9ec09d4eb44617fd0043680cdcd
```

逐图配对结果（`pp` 为百分点）：

| map | baseline explored | full explored | delta | baseline return | full return |
|---|---:|---:|---:|---:|---:|
| `112.png` | 25.69% | 23.42% | -2.27 pp | -30.74 | -31.76 |
| `119.png` | 22.76% | 9.36% | -13.40 pp | -29.62 | -32.31 |
| `121.png` | 19.14% | 9.95% | -9.19 pp | -29.56 | -45.13 |
| `124.png` | 40.27% | 14.88% | -25.40 pp | -26.89 | -36.70 |
| `13.png` | 15.27% | 14.30% | -0.97 pp | -31.65 | -31.89 |
| **mean** | **24.63%** | **14.38%** | **-10.25 pp** | **-29.69** | **-35.56** |

补充均值：

```text
success:               0.0 vs 0.0
steps:               128.0 vs 128.0
travel distance:     517.76 vs 570.45  (full +52.69)
repeated edges:      117.6  vs 125.0   (full +7.4)
planning latency:     11.61 vs 15.00 ms (full +3.39 ms)
```

5 张图都没有 explored-rate 改善。样本是列表前 5 张而非随机抽样，不能给出统计结论，但已经足够触发“停止扩算并先诊断”的工程决策。

`paths.jsonl` 和 `paired_preview.png` 只能作定性参考：collector 保存的是每步 `state.node_xy`，图状态可能重新以机器人为原点或随层级压缩变化，不应未经验证就把它当作绝对世界坐标轨迹。`124.png` 还出现 `repeated_edges=123`、`backtracks=1` 的反常组合，应优先审计行为指标定义和 stable-ID 语义。

## 6. 为什么更像算法负迁移，而不是训练故障

已经排除或没有证据支持：

- NaN/Inf；
- CUDA OOM、NCCL/DDP 故障；
- checkpoint/replay 不一致；
- Actor、Q 或 alpha optimizer 没有更新；
- 双 Q 意外相同；
- KF variance 非正；
- 标签退化为常数；
- 标签缓存改变数据语义。

仍未排除：

- full 的融合残差压过 baseline pointer logits；
- 未受监督或尚未充分受监督的 potential 被每状态标准化后放大；
- potential head 同时承受 SAC policy gradient 和 future-gain 监督，语义互相污染；
- KF posterior、stable region ID 或 event/retire 逻辑使候选排序恶化；
- hierarchy/diffusion 改变表征或动作槽映射；
- 训练地图趋势与固定 IID-test 行为严重不一致；
- baseline/evaluator 本身存在 success-floor、坐标、指标或环境设置问题。

## 7. 必须先理解的代码事实

`full` 相对 `ariadne_pi` 一次性同时打开了五个开关（potential/ranking、temporal、diffusion、hierarchy 四类机制）：

```yaml
potential: true
ranknet: true
temporal: kf
graph_diffusion: true
hierarchy: true
```

所以当前 5 图结果不能把问题归因到单个模块。

关键实现路径：

- `src/AC-PBGRL/ac_pbgrl/models/policy.py`
  - `_masked_standardize()` 会把每个状态中的 potential mean 强制变成单位尺度；即使预测仍是噪声，也可能获得与成熟特征相当的影响力。
  - `fusion` 的最终残差没有显式 gate、幅值上界或相对 `base_logits` 的尺度约束。
  - fused logits 从训练第一个 update 起就参与 SAC。
- `src/AC-PBGRL/ac_pbgrl/learning/sac.py`
  - offline future-gain auxiliary loss 在 update 10,000 前权重严格为 0，之后到 30,000 updates 才完全 ramp。
  - 200k checkpoint 仅为 update 12,375，auxiliary weight 只有 `0.04479/0.2`。
  - 即使 auxiliary weight 为 0，potential head/fusion 仍通过 policy logits 接收 SAC actor gradient。因此“future-gain 预测头”在受到足够监督前已经能改变策略，而且它的语义可能被 RL objective 改写。
- `src/AC-PBGRL/ac_pbgrl/learning/rollout.py`
  - 每步先以 raw potential 推理，再用 KF posterior 装饰状态并第二次 rescoring。
  - `collect()` 开头确实调用 `self.temporal.reset()`；因此不能简单声称 KF 跨地图泄漏，但 stable-ID、event、retire_missing、posterior 更新仍需逐项验证。
- `src/AC-PBGRL/ac_pbgrl/models/context.py` 与 `envs/ariadne/adapter.py`
  - hierarchy 声称保持 action slot 顺序，并通过 `candidate_old_indices` 映射回原 waypoint；需要补非连续 mask、重复/动态候选和真实地图的端到端断言。
- `src/AC-PBGRL/ac_pbgrl/learning/future_gain.py`
  - 每个有效候选会 clone 环境并执行最多 6-step teacher rollout；必须证明 clone 不共享会被 step 修改的内部状态。
- `src/AC-PBGRL/ac_pbgrl/evaluation/evaluator.py`
  - 当前全部结果最后统一写盘，没有逐图 checkpoint/resume。

## 8. 新窗口的优先诊断顺序

### P0：先验证比较是否可信

1. **baseline sanity**：解释为什么固定 IID-test 100/100 都跑满 128 步且 success=0。核对训练/评测 config、map split、start pose、completion threshold、reward、地图尺寸/难度及 published ARiADNE 行为。
2. **action-slot round trip**：从 actor 的 argmax slot，经 hierarchy compaction metadata，到 adapter 原始 waypoint，逐状态断言动作身份不变；覆盖非连续 candidate mask 和动态候选。
3. **clone isolation**：对真实 ARiADNE adapter 做 labeler 前后深状态哈希/观测一致性测试，证明 `environment.clone().step()` 不修改主环境。
4. **metric/path correctness**：复核 repeated edge、backtrack、travel distance 和 trajectory 坐标定义，特别解释 `124.png` 的反常组合。
5. **evaluation determinism**：设置 `CUBLAS_WORKSPACE_CONFIG`，同设备重复极小行为检查，确认 argmax 不因近似 tie 漂移。

### P1：用现有 checkpoint 做归因，不先重训

为 actor 增加只用于诊断的可控路径，并记录每个状态：

- `base_logits` 的均值/标准差/范围；
- fusion residual 的均值/标准差/范围及其与 base-logit scale 的比值；
- raw action mean、KF posterior mean、log variance；
- `argmax(base)`、`argmax(raw-fused)`、`argmax(KF-fused)` 的分歧率；
- 各 argmax 对应的 teacher future-gain target（仅诊断状态）；
- 候选数、stable ID、动作槽和实际 waypoint。

建议的 inference-only 模式：

1. full encoder + `base_logits` only；
2. raw potential fusion，不使用 KF posterior；
3. KF posterior + fusion（当前 full）；
4. fusion mean-only；
5. fusion uncertainty-only；
6. 必要时再隔离 diffusion 与 hierarchy。

这些模式应共享同一个 200k checkpoint，先回答“是哪一层造成决策变化”，不要立即启动多份训练。

### P2：确认归因后再选择算法修复

优先考虑但不要未经测量直接落地：

- 给 fusion 增加显式、可记录的 bounded gate，例如零初始化 gate 加有界 residual，保证初始策略严格退化为 base pointer；
- gate 随 potential 的 held-out ranking/calibration 或 auxiliary schedule 增长，而不是从第一个 SAC update 就全量生效；
- 避免对尚不可信的 potential 做无条件 per-state unit standardization；
- 分离“future-gain 语义头”和“策略残差”的梯度所有权，测试是否需要对 fusion 输入 detach，避免 SAC gradient 改写监督目标；
- 先离线预训练/验证 potential，再解冻策略融合；
- 对 residual/base-logit scale 加监控和保护；
- 只有 KF 的 raw potential 已可靠时，才让 temporal posterior影响动作排序。

## 9. 评测工具应先修的工程问题

在任何下一轮长评测前：

1. behavior metrics 与昂贵 potential diagnostics 分成两个明确阶段；
2. 每张地图结束后原子追加结果或写独立 shard；
3. 保存完成 map_id、seed、checkpoint SHA、config hash，并支持 resume；
4. 失败/中止后保留已完成地图，不重复数小时工作；
5. 输出实时进度和预计剩余地图数；
6. behavior-only 配对评测必须保持相同 map/seed/device/protocol；
7. potential diagnostics 可以在协议明确的确定性子集运行，但不得静默改变论文协议；
8. `paths.jsonl` 若用于论文轨迹图，应保存真实世界坐标或明确坐标系，而不是含义不稳定的 graph-local 坐标。

## 10. 建议的新窗口工作计划

1. 先读本文件、`AC_PBGRL_TRAINING_HANDOFF.md`、`PAPER_RESEARCH_ROUTE.md`、`src/AC-PBGRL/docs/EXPERIMENT_PROTOCOL.md`。
2. 检查本地 Git 状态，保留用户改动；当前预期分支为 `dev_kf`。
3. 阅读第 7 节列出的实现文件和已有测试，不连接服务器就先形成数据流图与可证伪假设。
4. 添加 P0 correctness tests；任何失败先修 correctness，不调算法权重。
5. 添加 logit-decomposition/inference-ablation 诊断接口，默认关闭，保证旧配置行为不变。
6. 修 evaluator 的逐图落盘与 resume，并为中止恢复添加测试。
7. 运行本地测试并汇报：根因证据、最小修复、行为不变断言、下一次最小服务器验证预算。
8. 只有用户批准后，先运行极小、同图、behavior-only 配对诊断；结果不改善则继续本地修复，不进入 500k。
9. 小样明确改善后，再由用户决定是否重新训练到 200k/500k；仍不自动进入多 seed、1M 或完整消融。

## 11. 下一版算法的最低验收门槛

在投入长训练前至少满足：

- gate=0 时，full policy 的动作与规定的 base-policy 路径逐状态一致；
- hierarchy on/off 的原始 action slot/waypoint 身份保持可验证；
- labeler clone 对主环境零副作用；
- potential 在 held-out labels 上 ranking 明显高于随机，且 top-1 regret 有实际意义；
- fusion residual 不无界压过 base logits；
- raw→KF posterior 的 argmax 改变可由 future-gain 改善解释；
- 先前 5 张失败地图至少不再出现系统性 explored-rate 回退；
- behavior evaluator 可逐图恢复，长评测中断不再丢失全部进度；
- 规划延迟和内存增量被明确量化；
- 所有结论仍标注为单训练 seed 的方向筛选，不冒充论文统计证据。

## 12. 本地开发与验证约定

关键入口：

```text
src/AC-PBGRL/configs/experiments/ariadne_pi.yaml
src/AC-PBGRL/configs/experiments/full.yaml
src/AC-PBGRL/ac_pbgrl/models/policy.py
src/AC-PBGRL/ac_pbgrl/models/potential.py
src/AC-PBGRL/ac_pbgrl/models/temporal.py
src/AC-PBGRL/ac_pbgrl/models/context.py
src/AC-PBGRL/ac_pbgrl/learning/sac.py
src/AC-PBGRL/ac_pbgrl/learning/rollout.py
src/AC-PBGRL/ac_pbgrl/learning/future_gain.py
src/AC-PBGRL/ac_pbgrl/envs/ariadne/adapter.py
src/AC-PBGRL/ac_pbgrl/evaluation/evaluator.py
src/AC-PBGRL/tests/
```

常规检查：

```bash
git status --short --branch
git log -8 --oneline
git diff --check
python3 -m py_compile src/AC-PBGRL/ac_pbgrl/cli.py
pytest -q src/AC-PBGRL/tests --disable-warnings --maxfail=1
```

编写本文件前的最新测试记录：

- 本地：28 passed、1 skipped（宿主 Python 无 PyTorch导致相关测试跳过）；
- 远端 CPU-only：49 passed；
- full label decoded-shard cache 在真实标签上已做逐元素语义一致性验证；
- 缓存吞吐从约 `3.666 s/update` 改善至稳定约 `1.96 s/update`，约 `1.84×`；该优化与当前行为负迁移应分开处理。

所有源码修改必须先在本地用 `apply_patch` 完成并测试，再 commit/push；远端只部署经过校验的文件。不要提交服务器凭据、大型 checkpoint/replay/labels、preview 图片或远端缓存。

## 13. 重要提交索引

```text
245a0d64 docs: record full 200k audit
dde504da docs: record label cache validation
90705256 perf: cache decoded future-gain label shards
7233e4ae perf: skip zero-weight offline potential batches
4b89c704 fix: deterministic no-replacement calibration sampling
53f80932 keep single-run figures descriptive
c3c64d75 lock the single-run paper-driver plan in regression tests
```

更早的 watchdog、DDP、GPU allowlist 和标签 worker 修复见 `AC_PBGRL_TRAINING_HANDOFF.md`。

## 14. 新窗口交付物定义

新窗口本阶段的完成标准不是“重新把训练跑起来”，而是：

1. 给出有代码/测试/已有产物支持的负迁移根因排序；
2. 修复已确认的 correctness 或评测恢复问题；
3. 提供默认不改变旧行为的诊断接口；
4. 提出一个最小算法改动，并解释为什么它能避免当前 full 的早期策略污染；
5. 通过本地测试；
6. 向用户提交下一次小规模验证的精确命令、地图数、资源、预计时长和停机条件，等待授权。

在上述工作完成前，不要恢复 cron，不要续训到 500k。
