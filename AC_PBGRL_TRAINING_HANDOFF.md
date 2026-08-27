# AC-PBGRL 论文训练与监控 Handoff

最后更新：2026-08-27 23:52（Asia/Shanghai）
当前分支：`dev_kf`  
核心实现基线：`53f80932`
最新流水线计划测试：`c3c64d75`
当前工作范围：只关注真实论文算法、二维训练、配对评测与实验可靠性；PPT 工作已经结束。

## 1. 本轮任务是什么

目标是在独立项目 `src/AC-PBGRL` 中训练和验证 AC-PBGRL，并持续看护远端训练。出现可复现的代码、资源调度、checkpoint 或评测故障时，应在不影响其他用户进程的前提下修复、测试、同步并从最近完整 checkpoint 恢复。

AC-PBGRL 的论文主线为：

1. 以 ARiADNE+PI 为基础，即 belief Actor、训练期 privileged GT Critic、离散 SAC，以及不改变索引的原始邻居 Pointer 动作空间。
2. 使用冻结 teacher 和 GT 有限视距 rollout 生成 action-conditioned future-gain 标签。
3. 增加 action-conditioned potential head，输出均值和有界方差；使用 Gaussian NLL 与 RankNet 辅助监督。
4. 将潜力 posterior 和 edge/path 特征真正送回 Pointer scorer，而不是只做不影响策略的旁路辅助头。
5. 使用校准后的 event-gated adaptive KF 跟踪稳定 region ID 的潜力记忆。
6. 使用 action-preserving 的 1/2/4-hop graph diffusion、局部节点和远端 region token；原始候选动作及其索引始终保留。

详细论文路线见 `PAPER_RESEARCH_ROUTE.md`，实现与运行说明见 `src/AC-PBGRL/README.md`，公平评测协议见 `src/AC-PBGRL/docs/EXPERIMENT_PROTOCOL.md`。

## 2. 为什么当前先跑 pilot，而不是直接跑完整论文实验

正式主对比原计划是每种方法 `5 seeds × 1M transitions`。含义是：每种方法独立训练 5 个模型，每个模型经历 1,000,000 条环境 transition；每个 seed 都有不同的初始化、探索轨迹和 replay 采样。它不是同一个模型评测 5 次。

在尚未确认完整方法确有收益之前，直接投入完整主对比和消融会消耗数周。用户已明确要求当前先让每个对比版本只跑一次，因此现阶段是单次方向筛选，不是多种子可信 pilot：

- 只比较 `ariadne_pi` 与 `full`；
- 每种方法只训练 seed `0` 一次；
- 每个 run 先到 200k transitions，生成早期诊断；
- 随后从同一 checkpoint 无损续训到 500k transitions，再判断优化方向是否值得投入多种子；
- 每个 checkpoint 使用相同的 IID map 子集和 seed 做 100-map 配对评测；
- teacher、标签、baseline 与 full 的 checkpoint、replay 都可被后续正式 1M 流程继续复用。

单个训练 seed 只能发现实现故障、明显负迁移和描述性的配对地图改善，不能估计训练随机性，也不能支持论文统计结论。代码要求显式提供 `--single-run-screening --pilot-seeds 1`，并在结果 manifest 标记 `single_run_directional_screening`；未提供该开关时，可信 pilot 仍强制至少 3 个独立 seed。

不能把 100k 当作有效的优化效果验证点。正式配置为：

```text
loss.warmup_steps = 10,000 optimizer updates
loss.ramp_steps = 20,000 optimizer updates
train.gradient_updates_per_transition = 0.0625
```

所以：

```text
warm-up 结束：10,000 / 0.0625 = 160,000 transitions
辅助权重完全增至目标值：(10,000 + 20,000) / 0.0625 = 480,000 transitions
```

200k 只用于发现发散、NaN、策略坍缩或明显负迁移，不应据此宣称方法有效或调参追结果。500k 是不改变正式超参数时，能完整经历辅助损失 schedule 的最小可信训练预算。

当前筛选训练量为 `2 methods × 1 seed × 500k = 1M transitions`，另有已经完成的 100k teacher、20k/2,048 个 train/validation 标签及两次配对评测。应在第一个完整 200k run 后用实际吞吐重新估计总时间。

## 3. 当前无人值守流水线的精确顺序

当前 watchdog 执行的逻辑等价于：

```bash
./run.sh paper --pilot-only \
  --single-run-screening \
  --pilot-seeds 1 \
  --pilot-early-steps 200000 \
  --pilot-steps 500000 \
  --gpus auto \
  --gpu-policy prefer-idle
```

流水线顺序为：

1. 将共享的 `ariadne_pi` teacher 训练到 100k；若已有 checkpoint 则自动恢复。
2. 创建或复用 leakage-safe map split。
3. 生成/续生成 20,000 个 train future-gain 标签和 2,048 个 validation 标签。
4. 将 `ariadne_pi/seed_0` 训练到 200k。
5. 对 `full/seed_0` 先完成 30k temporal/KF calibration prephase，再训练到 200k。
6. 在固定单卡上对两个 checkpoint 做相同 100-map IID 配对评测，写入 `pilot/step_200000`，并绘图。
7. 将 `ariadne_pi/seed_0` 从 200k 续到 500k。
8. 将 `full/seed_0` 从 200k 续到 500k。
9. 用 validation 标签重新校准 full checkpoint 的 uncertainty/KF temperature。
10. 再做相同的配对评测，写入 `pilot/step_500000`，绘图并原子写 single-run manifest。
11. watchdog 写入 `pilot_pipeline.complete` 后退出；它不会自动进入完整 5-seed/1M 正式实验。

pilot 的运行名与正式运行名相同，例如 `runs/full/seed_0`。如果 pilot 通过，后续正式流程会从 500k 继续到 1M，而不是重练前 500k。

## 4. 当前服务器状态快照

本节是 2026-08-27 23:52 左右的快照，PID 和数值会变化，接手后必须重新查询，不可依赖这里的 PID。

- 共享 `ariadne_pi` teacher 已完成并固化到 100,000 environment transitions、6,125 optimizer updates、807 episodes；checkpoint 与 replay 的 `total_added` 都是 100,000。
- 当前 stage：train/validation future-gain 标签均已完成并通过最终审计；`ariadne_pi/seed_0` 正在向 200k transitions 训练。23:50:54 的第八个完整 checkpoint 为 50,533 environment transitions、400 episodes、3,033 optimizer updates；随后第九轮 rollout 已把磁盘 replay 推进到 56,607，正在执行对应 update wave。
- train manifest 已完成 20,000 samples / 20 shards（19 × 1,024 + 544）；最终聚合审计确认全部 325,820 个有效候选标签、node/edge features 与 mask 均通过完整性检查，teacher/map-split provenance 哈希一致。
- 最终 train 标签分布未退化：均值 `-0.832`、标准差 `0.761`、范围 `[-3.746, 2.913]`，按六位小数约有 208,225 个唯一值。
- validation manifest 已完成 2,048 samples / 2 shards，共 33,414 个有效动作标签；全部 510 张 validation 地图均来自 validation split，与 train/OOD/IID-test 的地图重叠为 0，provenance 与 train 完全一致。
- 修复后的首 shard（包含 Ray 初始化）耗时约 11 分钟，即约 1.53 samples/s；据此 train 20k 粗估约 3.6 小时，validation 粗估约 22 分钟，应以后续 shard 实测继续校正。
- watchdog ID：`pilot`。
- 调度硬限制：`ACPBGRL_GPU_ALLOWLIST=0,3`。
- 标签阶段是 CPU-only，GPU 0/3 当时空闲；当前训练仍只允许使用物理 GPU 0 和 3，两张卡在更新时各约占 27.5 GiB。
- GPU 1 和 2 上存在其他用户的约 12 GiB 进程，本项目不得探测训练显存、租用或启动在这两张卡上。
- 标签阶段的 label driver 与 32 个 Ray `LabelWorker` 已正常退出；最终标签产物已完整固化。
- 标签 Ray runtime 只发布了 32 CPU、0 GPU；worker niceness 为 0，affinity 为 `0-95`，未再出现全部拥挤在 `0,48` 的问题。
- watchdog heartbeat 每 30 秒更新，且 cron 同时配置 `@reboot` 与每分钟兜底调用，因此本地 terminal、SSH 或 Codex 对话关闭不会终止训练。
- 当前 `ariadne_pi/seed_0` launch 使用物理 GPU `[0, 3]`、DDP world size 2、micro-batch 64 和 48 个 Ray rollout worker；两个训练 rank 的 `CUDA_VISIBLE_DEVICES`、`LOCAL_RANK`、`RANK` 与 `WORLD_SIZE` 已逐进程核对。
- 前两个完整训练周期均已验证：8,115 transitions 对应 382 updates，14,259 transitions 对应 766 updates，精确满足扣除 2,000 条 minimum replay 后每 transition `0.0625` 次更新的累计预算；两个 checkpoint 都能重新加载，第二个 checkpoint 的 replay 状态与当时磁盘状态完全一致。
- 25% checkpoint 可重新加载，checkpoint 与 replay 的 `total_added=50,533` 完全一致，SHA256 为 `46c2240e42abf0f06b5085e4acc44d33664a0a43776a2de7a5a69fab15657108`；截至该 checkpoint 的 408 条 train metrics 所有数值字段均为有限值，没有 NaN/Inf、CUDA OOM、NCCL 故障或 watchdog 重启。
- 25% 最新 aggregate 为 Q loss `37.48/37.65`、Q gradient `218.40`、target `70.32`、entropy `2.604`、alpha `0.753`，仍处于已稳定完成 100k 的 teacher 轨迹包络内。全部 400 个在线训练 episode 成功率 4.0%；最近 192 个 episode 成功率 5.2%、平均覆盖率 0.581、平均回报 -48.99，未出现策略坍缩；这些训练地图噪声样本不可提前当作固定地图效果评测。
- 远端发布目录不带 `.git` 元数据，但 `cli.py`、`learning/train.py`、`learning/sac.py`、`runtime/supervisor.py` 与 `server_watchdog.sh` 五个关键文件的 SHA256 已和本地逐项核对完全一致，后续 full 阶段不会加载漂移源码。
- update wave 时 GPU 0/3 各占约 27.5 GiB，利用率通常 90%～100%，温度约 74/78°C、功耗约 250W，未出现硬件或软件 thermal throttle；rollout/checkpoint 边界的单卡瞬时低利用率不代表训练停止。
- supervisor 运行中每 30 秒检查资源压力；任一已选 GPU 连续 2 次严格高于配置上限 80°C 时，会写请求文件并等待当前 update 边界优雅 checkpoint/重启，而不是直接杀训练。单次读数等于 80°C 不触发；硬件 slowdown 阈值为 95°C。若发生重复温度重启，应先核对风道、外部 GPU 进程和 event 记录，再决定是否降低 micro-batch 或功耗，不能手工 kill rank。
- 远端完整测试：45 passed。
- 本地测试：28 passed、1 skipped；跳过原因是本地宿主 Python 没装 PyTorch，不是代码失败。
- `c3c64d75` 增加了端到端 mocked paper-driver 计划测试，锁定单次模式的精确顺序，并明确断言不会启动 seed `1` 或消融 run。

远端连接地址、用户名、密码和私有绝对路径按项目约束不写入 Git。新会话应从用户提供的连接信息或本机 SSH config 获取；若信息不可见，应重新向用户索取。禁止把认证信息加入命令脚本、README、handoff、Git history 或日志。

连接后，可通过服务器的 `crontab -l` 找到已经安装的绝对项目目录、数据根目录和 watchdog 命令。下文统一使用：

```bash
REMOTE_PROJECT_ROOT=<远端 AC-PBGRL 源码目录>
ACPBGRL_DATA_ROOT=<远端 /mnt 持久化数据目录>
ACPBGRL_PYTHON=<远端独立环境中的 python>
```

## 5. 新窗口接手后的第一组检查

先在本地仓库执行：

```bash
git status --short
git branch --show-current
git log -8 --oneline
```

预期分支是 `dev_kf`，handoff 编写时核心实现已推送到 `53f80932`。若工作树出现其他改动，先判断是否为用户改动，不要覆盖或回滚。

连接训练服务器后执行只读检查：

```bash
crontab -l
cat "$ACPBGRL_DATA_ROOT/orchestration/pilot_watchdog.pid"
cat "$ACPBGRL_DATA_ROOT/orchestration/pilot_watchdog.heartbeat"
tail -n 30 "$ACPBGRL_DATA_ROOT/orchestration/pilot_watchdog.log"
tail -n 60 "$ACPBGRL_DATA_ROOT/orchestration/pilot_driver.log"
pgrep -af 'server_watchdog|ac_pbgrl.cli paper|ac_pbgrl.cli supervise|torch.distributed.run|ac_pbgrl.learning.train'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
```

检查要点：

- heartbeat 时间距当前时间不应超过约 90 秒；
- 当前 training launch 的 `gpu_indices` 只能是 `[0]`、`[3]` 或 `[0, 3]`；
- 不要只看整个历史 `events.jsonl` 搜索 GPU 1/2，因为切换事故曾留下历史记录；应看最新 launch 的时间与当前 PID；
- 当前 update wave 时 GPU 显存约 27.5 GiB、利用率接近 100% 是正常状态；rollout/Ray 初始化期可能只有约 0.6 GiB，并出现一个 rank 等待另一个 rank；
- GPU 1/2 上的 PID 不是本项目所有，绝对不能 kill；
- checkpoint 时间、metrics/replay 的 transition 数和进程状态至少有一项应持续推进；训练设计允许 replay/在线 rollout 暂时领先最近 checkpoint 一个 wave。

读取 checkpoint 进度可在远端使用：

```bash
"$ACPBGRL_PYTHON" - <<'PY'
import os
from pathlib import Path
import torch

root = Path(os.environ["ACPBGRL_DATA_ROOT"])
path = root / "runs/teacher_training/ariadne_pi/checkpoints/latest.pt"
payload = torch.load(path, map_location="cpu", weights_only=False)
print(payload["learner"]["learner_state"])
print("replay_total_added", payload["replay"]["total_added"])
PY
```

进入后续 run 时，应把相对路径替换为当前 `runs/<method>/seed_<n>/checkpoints/latest.pt`。当前 run 可从 process command、supervisor 最新 event 或最近修改的 metrics 文件判断，不要硬编码。

## 6. 持续监控的文件与判据

所有路径均相对于 `$ACPBGRL_DATA_ROOT`：

```text
orchestration/pilot_watchdog.pid
orchestration/pilot_driver.pid
orchestration/pilot_watchdog.heartbeat
orchestration/pilot_watchdog.log
orchestration/pilot_driver.log
supervisor/<run_name>/events.jsonl
runs/<run_name>/metrics/train.jsonl
runs/<run_name>/checkpoints/latest.pt
runs/<run_name>/run_manifest.json
replay/<run_name>/metadata.json
labels/train/
labels/validation/
pilot/step_200000/
pilot/step_500000/
```

每次状态检查至少确认：

1. watchdog 和 driver 活着，heartbeat 新鲜；
2. 当前 stage/run/seed 与计划顺序一致；
3. checkpoint 或 metrics 的 environment transitions 正常增加；
4. loss、gradient、alpha、entropy 中没有 NaN/Inf 或持续爆炸；
5. replay metadata 与 checkpoint 能对应，重启后没有倒退到不一致状态；
6. 最新 launch 只使用 allowlist 内 GPU；
7. 显存保留和温度仍安全，没有 CUDA OOM 循环；
8. Ray worker 数、CPU load、内存与 `/mnt` 剩余空间没有成为瓶颈；
9. driver 日志没有重复 traceback；单次可恢复退出不等同于持续故障；
10. 完成 200k 或 500k 后，六个配对评测目录、CSV、potential samples、图和 manifest 全部存在。

建议检查磁盘：

```bash
df -h "$ACPBGRL_DATA_ROOT"
du -sh "$ACPBGRL_DATA_ROOT"/{runs,replay,labels,pilot} 2>/dev/null
```

不要删除 checkpoint、replay、labels 或实验产物来腾空间，除非用户明确授权并已确认可恢复目标。

## 7. 200k 和 500k 应如何解读

### 200k：工程与方向诊断

只判断：

- 是否能稳定训练和完成评测；
- potential 指标是否开始优于随机排序；
- full 是否发生明显策略坍缩；
- 是否存在动作索引、KF reset、calibration 或 diffusion/hierarchy 的实现错误；
- latency、节点规模或内存是否完全不可接受。

由于辅助权重尚未 fully ramped，不要根据 200k 的轻微输赢修改正式超参数，也不要把它当论文结果。

### 500k：单次方向筛选

优先看：

- exploration completion distance；
- 95% coverage distance；
- success/completion rate；
- makespan、最终覆盖率；
- backtracking、repeated edges、direction switches；
- planning latency、graph nodes、显存；
- potential RMSE/MAE/NLL；
- Spearman、Kendall、pairwise accuracy、top-1 regret；
- uncertainty calibration、coverage、KF NIS/reset/event 次数。

建议只做方向判断：

1. 比较同一 100-map 子集上的 paired distance 分布，确认改善不是只靠极少异常地图；描述性改善最好达到约 3% 以上；
2. success rate 不应出现超过约 2 个百分点的明显回退；
3. potential ranking 必须明显高于随机，并且 uncertainty calibration 不能完全失真；
4. 规划延迟、内存和节点预算仍满足可部署范围；
5. 收益应能通过代表路径与行为指标解释，而非只有 reward 上升。

100 maps 只是同一训练 seed 上的配对 episode，不能冒充独立训练重复。单次筛选可以否决明显无效或不稳定的设计，但不能确认方法普遍有效。只有方向值得继续时，才由用户决定启动 3-seed pilot 或正式 5-seed/1M；正式统计仍需按 seed 和 map 的层级/cluster 结构处理。

单次模式的 `paired_effects.csv` 只保留描述性地图配对差值、effect size 和 bootstrap 区间，不输出 Wilcoxon/Holm p 值；`figure_manifest.json` 必须记录 `inferential_statistics_included=false`、`statistical_claims_supported=false` 和 `map_variation_conditional_on_one_training_seed`。这是为了防止把 100 个 map episode 误当成 100 个独立训练重复。

## 8. 已修复的重要故障与历史日志陷阱

相关提交均已推送：

```text
c04a829a  request graceful DDP stops without Unix signals
c164ddcd  initialize zero-transition stop branch
1537bb65  cleanly exit coordinated training stops
4682e696  prioritize main comparison before ablations
2cfc111a  synchronize graceful DDP shutdown decisions
0b5bee53  add minimum credible pilot pipeline
ac916582  restrict scheduler to approved GPU indices
053a38b2  keep watchdog attached during graceful stop
5df4f15d  restore CPU scheduling for label workers
abf8a710  add explicit single-run screening mode
53f80932  keep single-run figures descriptive
c3c64d75  lock the single-run paper-driver plan in regression tests
```

### DDP graceful-stop rank race

旧逻辑可能让 rank 0 删除/归档 request 后，rank 1 才检查文件，从而一边返回 0、一边返回 75，torch elastic 报 `ChildFailedError`。`2cfc111a` 改为只由 rank 0 观察 stop，再广播统一决定。服务器 CPU DDP smoke 与完整测试已经通过。

### watchdog 收到信号后过早脱离 child

Bash 的 `wait` 会被 trap 中断，即使 paper driver 仍活着。旧 watchdog 因此可能退出并把 driver 变成 orphan，cron 又启动第二条流水线。`053a38b2` 让 watchdog 在 child 真正退出前持续 wait，并重复传递停止请求。

### 切换 full paper → pilot 时的短暂重叠

在修复 watchdog 前的这次切换中，旧 driver 曾短暂 orphan，而新 pilot watchdog 被 cron 启动。两个 supervisor 指向同一个 teacher run，导致：

- 历史 `events.jsonl` 中出现过 GPU 1/2 的 memory probe/launch 记录；
- torchrun 曾出现一次 `EADDRINUSE`；
- request file 曾被另一 supervisor 归档；
- pilot watchdog 在 17:18～17:20 有数次退出/重试记录。

处理结果：

- 用临时完成标记阻止 cron 重叠；
- 恢复 request file，让旧 DDP 在更新边界保存；
- 完整 checkpoint 保存到 74,164 transitions；
- 所有旧进程和临时 pilot 进程已退出；
- 增加并启用 `ACPBGRL_GPU_ALLOWLIST=0,3`；
- 部署修复后的 watchdog；
- 移除临时标记，并重新启动唯一 pilot watchdog；
- 当前最新 launch 已明确记录 `gpu_indices: [0, 3]`，训练已继续推进。

因此不要把旧日志中的 `gpu_indices: [1]`、`[2,1]` 或旧 `EADDRINUSE` 当成当前故障。判断是否复发必须结合最新时间戳、当前 PID 和连续三次以上的相同失败。

### 可忽略的已知警告

服务器会打印：

```text
expandable_segments not supported on this platform
```

这是当前 CUDA allocator/platform 的兼容警告；micro-batch 真实探测仍通过，且 A40 update wave 稳定占用约 27.5 GiB。目前不需要因此中止训练。

### 标签 worker 继承窄 CPU affinity

旧 `RayLabelPool` 没有训练 rollout pool 已有的 CPU affinity 恢复逻辑，导致 32 个标签 worker 都继承 `0,48`，同时保留 Ray 默认 niceness 15。机器虽然有 96 个逻辑 CPU，标签生成实际只获得约 1～2 核吞吐，首个 shard 长时间不能落盘。

`5df4f15d` 已修复：标签 Ray runtime 只声明 `label_actors` 个 CPU 和 0 GPU、禁用 worker niceness，并在每个 actor 初始化时把 affinity 恢复到可见 CPU 集。部署时按安全暂停流程停止旧 watchdog，远端 CPU-only 全套测试 43 passed 后恢复原 cron；新 worker 已实测 niceness 0、affinity `0-95`，单 worker CPU 从约 5% 提升到约 70%～85%。首个 1,024-sample shard 及 manifest 已正常写出。

## 9. 资源使用约束

- 只允许自动调度 GPU 0/3；保持 `ACPBGRL_GPU_ALLOWLIST=0,3`。
- 不要为了凑更多卡切换成 GPU 1/2，也不要结束其他用户进程。
- `prefer-idle` 会在 allowlist 内动态选 1～2 张卡。
- 两卡时当前真实 memory probe 选择 micro-batch 64，固定 global batch 128。
- GPU 数变化以 checkpoint 驱动的 supervisor 重启实现，不在 DDP 进程组内热增减。
- 训练公平性按 environment transitions、optimizer updates 和固定 global batch 计量，而不是墙钟时间。
- CPU rollout 当前会启动约 48 个 Ray actor，并恢复 worker CPU affinity；不要随意增加到耗尽系统内存或影响其他用户。
- 正式规划延迟必须在固定单卡重新评测，不能比较训练时不同卡数下的 latency。

## 10. 安全暂停与恢复

因为 cron 每分钟会补拉 watchdog，只 kill 当前 PID 会被自动重启。需要人工暂停时：

1. 先临时禁用或注释 AC-PBGRL 的两条 cron watchdog 项；
2. 从 `pilot_watchdog.pid` 读取当前 PID；
3. 向该 watchdog 发送 `SIGTERM`，不要直接 kill DDP rank；
4. 等待 supervisor 在更新边界写入 `interrupted.pt/latest.pt` 并确认整棵本项目进程树退出；
5. 检查 checkpoint 的 learner state 与 replay metadata；
6. 恢复时重新启用 cron，保持相同 watchdog ID、allowlist 和 pilot 参数。

示意：

```bash
pilot_pid="$(cat "$ACPBGRL_DATA_ROOT/orchestration/pilot_watchdog.pid")"
kill -TERM "$pilot_pid"
```

只允许信号发送给从本项目 PID 文件解析且经 `ps` 验证过的 PID。禁止按 `python` 名称批量 kill，禁止影响 GPU 1/2 的外部 PID。

若 CUDA OOM，supervisor 会从最新完整 checkpoint 恢复并降低 micro-batch。若同一故障连续复现，应保存：最新 traceback、supervisor events、run manifest、checkpoint learner state、GPU 快照和复现命令，然后在本地修复并测试；不要通过删除 replay 或重新初始化模型掩盖问题。

## 11. 本地修改、测试和远端同步约定

- 本地仓库是 Git 真源；远端训练源码目录当前不是 Git repository。
- 所有源代码改动先在本地使用 `apply_patch`，运行相关测试，再 commit/push。
- 远端离线，必须从本地传输精确改动文件或校验过的归档。
- 同步代码不会改变已经运行的 Python 进程；需要新代码生效时，必须走 checkpoint-aware graceful restart。
- 不提交 wheelhouse、模型、replay、labels、缓存、实验大文件、服务器地址或凭据。
- 不回滚用户其他改动，不用 `git reset --hard`，不删除不明确的远端数据。

常规验证：

```bash
git diff --check
python3 -m py_compile src/AC-PBGRL/ac_pbgrl/cli.py
bash -n src/AC-PBGRL/scripts/server_watchdog.sh
pytest -q src/AC-PBGRL/tests --disable-warnings --maxfail=1
```

远端完整测试应显式避免占用训练 GPU，例如设置空的 `CUDA_VISIBLE_DEVICES`，并在 CPU 资源允许时执行。部署 `abf8a710` 后最新结果为 44 passed。

## 12. 接下来应做什么

按优先级：

1. 持续观察 label generation 的 CPU/Ray 吞吐和 shard manifest，确认 train 从当前 6,144 达到 20k、validation 达到 2,048；若中断，应从最后完整 manifest 续写，不能删除已完成 shard。
2. 标签完成后确认流水线自动进入 `ariadne_pi/seed_0`，并重新检查 GPU allowlist、memory probe 和首个 checkpoint。
3. 监控 `ariadne_pi/seed_0` 到 200k，再监控 `full/seed_0` 的 prephase/calibration 和 200k 训练。
4. 200k 配对评测完成后，检查两份 episode CSV、potential samples、paired effects、代表路径和 single-run manifest；只做工程/方向诊断。
5. 流水线继续到 500k，不修改正式 warm-up/ramp 或数据 split。
6. 500k 完成后按第 7 节做单次方向分析，明确标注不能支持统计结论。
7. 若方向值得继续，再由用户确认先扩为 3-seed pilot，或启动正式主对比：ARiADNE、ARiADNE+PI、Q-distillation、full，各 5 seeds × 1M；现有两个方法的 seed 0 从 500k 续训。
8. 正式主对比通过后再投入完整消融，避免提前消耗计算。
9. 正式论文统计前审计 paired bootstrap/Wilcoxon 的层级独立性，并补 small-to-large、IID/OOD、latency 和失败案例。

不要因为一次日志不刷新就立即重启：Ray rollout 和一个完整 optimizer wave 都可能持续数分钟。通常应同时看 GPU、CPU、metrics、checkpoint 和事件；只有多项均停止推进且同一阻塞连续复现，才按故障处理。
