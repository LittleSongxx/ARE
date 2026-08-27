# AC-PBGRL

AC-PBGRL（Action-Conditioned Future-Potential Belief Graph RL）是一套可独立运行的二维自主探索论文代码。项目保留 ARiADNE 的真实邻居 Pointer 动作契约，并在 belief Actor / privileged GT Critic / discrete SAC 基础上加入：

- 训练期冻结 teacher 的六步 GT rollout future-gain 标签；
- action-conditioned Potential Head（region potential + action residual）；
- Gaussian NLL、RankNet 和 10k warm-up + 20k ramp-up；
- held-out 方差校准及 event-gated adaptive KF；
- no-memory、EMA、离线训练 GRU 三种时序对照；
- 1/2/4-hop graph diffusion、保留全部真实动作的局部图、远端 region token 与 A* context skeleton；
- 可变 1～4 GPU、固定全局 batch、checkpoint 驱动的安全重启；
- 配对统计、论文图表、ONNX 和 ROS Noetic waypoint 部署。

`src/AC-PBGRL` 不导入其他 `src/*` 项目，也没有符号链接。所需模拟器、图构建代码和地图均在本目录内。来源及再发布限制见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 方法边界

项目包含三个核心比较对象：

| 配置 | Actor | Critic | 新增机制 |
|---|---|---|---|
| `ariadne` | belief graph | belief graph | 无 |
| `ariadne_pi` | belief graph | privileged GT graph | 仅 asymmetric Critic |
| `full` | belief graph | privileged GT graph | potential distribution + calibrated KF + multiscale context |

`ariadne_pi` 是项目内公平控制基线，不声称完整复现 Cao et al. (2024) 论文中的 graph-rarefaction 工程链。TARE 只用于 ROS/Gazebo 物理指标轨，安装器锁定上游提交，不把它包装为本文贡献。

策略的数据流是：

```text
belief graph ──> 1/2/4-hop diffusion ──> action-preserving graph encoder
                                        ├─> base Pointer logits
current/candidate/edge/global context ──> Potential Head
                                        ├─> action mean / variance
stable candidate ID ──> calibrated KF ──> region posterior
                                        └─> posterior-aware Pointer logits

GT map + frozen teacher ──> finite rollout labels ──> NLL + RankNet（仅训练期）
GT graph ─────────────────────────────────> twin Q Critic（仅训练期）
```

Pointer 始终只在原始 `current_edge` 的动作 slot 上决策；region token 和 A* skeleton 只提供上下文，不会成为不可执行 waypoint。

## 快速入口

```bash
./run.sh doctor
./run.sh labels --config full --split train --samples 100000
./run.sh train --config full --gpus auto
./run.sh train --config full --gpu-policy idle-only
./run.sh supervise --config full --min-gpus 1 --max-gpus 4
./run.sh ablate --suite main --gpus auto
./run.sh evaluate --config full
./run.sh figures
./run.sh export --config full --format onnx --validate
./run.sh paper --gpus auto
```

`evaluate` 和 `export` 默认读取该配置/种子的 `checkpoints/latest.pt`，也可显式传入 `--checkpoint`。所有命令支持 `--set key=value`，例如：

```bash
./run.sh train --config potential_nll_rank --gpus cpu --smoke \
  --set project.data_root=/tmp/ac-pbgrl-smoke \
  --set project.seed=3
```

## 配置与消融

`configs/base.yaml` 固定共同的观测、动作、全局 batch 和正式 transition 预算。`configs/experiments` 只覆盖方法差异：

| 配置 | 用途 |
|---|---|
| `q_distillation` | GT Critic Q 直接蒸馏控制 |
| `potential_mse` | future-gain 均值回归 |
| `potential_nll` | 异方差 Gaussian NLL |
| `potential_nll_rank` | NLL + RankNet |
| `potential_kf` | potential + calibrated KF，无多尺度上下文 |
| `potential_diffusion` | potential + diffusion/context，无时序滤波 |
| `ema_control` | EMA 时序对照 |
| `gru_control` | 离线 GRU 时序对照 |
| `full` | 完整 AC-PBGRL |

正式主实验使用 5 个种子，消融使用 3 个种子。训练默认固定为 1,000,000 environment transitions、全局 batch 128 和每 transition 0.0625 次梯度更新；episode 数只是安全上限。这样 GPU 数量和策略完成 episode 的速度不会改变训练预算。

KF/GRU 方法需要先得到无时序预训练 checkpoint：

```bash
# KF：先关闭 temporal 训练，再用 validation labels 校准
./run.sh train --config full --set method.temporal=none --set train.max_environment_steps=30000
./run.sh calibrate --config full --checkpoint /path/to/latest.pt --split validation
./run.sh train --config full --resume auto

# GRU control：从稳定 ID 的离线标签序列训练
./run.sh train-gru --config gru_control --actor-checkpoint /path/to/latest.pt
```

`./run.sh paper` 会按 teacher → leakage-safe map split → train/validation labels → KF/GRU pre-phase → 主实验/消融 → 固定单卡评测 → figures 的顺序执行完整流程。

## 动态 GPU 与恢复

默认 `prefer-idle` 策略只要存在空闲卡，就不会为凑满四卡主动占用共享卡。三种策略为：

- `idle-only`：仅完全空闲卡；
- `prefer-idle`：空闲优先，无空闲卡时才考虑满足阈值的共享卡；
- `shared-ok`：可主动选择所有满足安全阈值的卡。

默认要求可用显存至少 18 GiB、利用率不高于 65%、温度不高于 80℃，并在进程内保留至少 6 GiB。启动前会执行真实 Actor + twin Critic + potential batch 的短前后向探测；A40 从 `64 → 32 → 16 → 8 → 4` 回退。固定全局 batch 128 对应：

| GPU 数 | 每 rank 样本 | A40 有效 chunk | 累计次数 |
|---:|---:|---:|---:|
| 4 | 32 | 32 | 1 |
| 3 | 43/43/42 | 43/43/42 | 1 |
| 2 | 64 | 64 | 1 |
| 1 | 128 | 64 | 2 |

CPU rollout 默认按 48 个物理核留出系统余量后扩展为 1/2/3/4 卡对应 32/48/56/64 个 Ray actor。worker 启动时会恢复完整 CPU affinity 并使用单线程算子，避免 OpenMP 启动线程把全部 Ray 子进程错误限制在一个物理核上。GPU 数、actor 数或 micro-batch 的变化都不改变统一的 environment-transition、optimizer-update 和 global-batch 预算。

supervisor 每 30 秒记录资源与外部 GPU 进程。显存余量连续不足或收到信号时，它通过运行目录中的 request file 通知 DDP ranks，在共同更新边界原子保存 checkpoint 后退出；这避免 Ray 原生 SIGTERM handler 和 torch elastic 提前杀死训练进程。CUDA OOM 仍由训练器写入专用恢复标记并降低 micro-batch。随后 supervisor 重新选卡续训。项目锁只协调 AC-PBGRL 自己的任务，不会结束或抢占其他用户进程。`run_manifest.json` 保留所有 4→2→1 卡资源会话，`supervisor/**/events.jsonl` 保存等待、压力、OOM 和重启事件。

### 无人值守与服务器重启恢复

`nohup` 足以让任务不受本地终端、SSH 断线或 Codex 会话结束影响；正式长实验建议再用外层 watchdog 覆盖 paper driver 崩溃与服务器重启：

```bash
ACPBGRL_DATA_ROOT=/mnt/songensheng/ac-pbgrl \
ACPBGRL_PYTHON=/mnt/songensheng/ac-pbgrl/env/bin/python \
nohup setsid ./scripts/server_watchdog.sh paper --gpus auto --gpu-policy prefer-idle \
  </dev/null >/dev/null 2>&1 &
```

watchdog 使用 `flock` 保证同名流水线只有一个实例，driver 非零退出后等待 30 秒再从原子 checkpoint/replay 续跑；成功完成后写入 `orchestration/paper_pipeline.complete` 并停止重启。PID、30 秒心跳和事件分别写入 `orchestration/paper_watchdog.pid`、`paper_watchdog.heartbeat` 和 `paper_watchdog.log`。可通过 `ACPBGRL_WATCHDOG_ID` 为不同流水线隔离锁、日志和完成标记。

共享服务器上若只获准使用部分物理卡，可设置如 `ACPBGRL_GPU_ALLOWLIST=0,3`。`auto` 调度仍能在这两张卡之间按资源动态缩放，但不会探测、租用或训练在名单外的 GPU。

在投入正式的 `5 seeds × 1M transitions` 前，可先运行最小可信预验证。它只比较 ARiADNE+PI 与完整 AC-PBGRL，使用 3 个独立训练 seed；在 200k transitions 生成一次早期诊断，并无损续训到 500k。正式配置中辅助损失在约 480k transitions 才完成 warm-up 与线性增权，因此 200k 结果只用于发现训练失稳，500k 才作为 pilot 的效果判断点：

```bash
ACPBGRL_WATCHDOG_ID=pilot \
ACPBGRL_GPU_ALLOWLIST=0,3 \
ACPBGRL_DATA_ROOT=/mnt/songensheng/ac-pbgrl \
ACPBGRL_PYTHON=/mnt/songensheng/ac-pbgrl/env/bin/python \
nohup setsid ./scripts/server_watchdog.sh paper --pilot-only \
  --pilot-seeds 3 --pilot-early-steps 200000 --pilot-steps 500000 \
  --gpus auto --gpu-policy prefer-idle </dev/null >/dev/null 2>&1 &
```

pilot 复用正式运行名与 checkpoint；若结果通过，后续正式 1M 流程从 500k 继续，而不是重新训练。配对评测与图表分别写入 `pilot/step_200000` 和 `pilot/step_500000`。

如果当前只需要检查优化方向、暂不投入多种子，可显式运行单次筛选。它仍完整经历 200k 诊断、500k auxiliary schedule 和相同的 100-map 配对评测，但只训练 seed 0：

```bash
./scripts/server_watchdog.sh paper --pilot-only --single-run-screening \
  --pilot-seeds 1 --pilot-early-steps 200000 --pilot-steps 500000 \
  --gpus auto --gpu-policy prefer-idle
```

单次筛选的 manifest 会标记 `single_run_directional_screening`，只能用于判断实现是否稳定、效果方向是否值得继续，不能替代论文级多种子统计。未提供 `--single-run-screening` 时，可信 pilot 仍强制至少 3 个独立 seed。

该模式的图表仍保留同一 seed 内的配对地图差值和描述性 bootstrap 区间，但不会输出 Wilcoxon/Holm p 值；`figure_manifest.json` 会把不确定性范围标记为 `map_variation_conditional_on_one_training_seed`，避免把地图 episode 误当成独立训练重复。

无 root 权限的服务器可把同一命令写入用户 crontab 的 `@reboot`，并额外每分钟调用一次作为 watchdog 自身的兜底；重复调用会因项目锁立即退出。安装前应使用绝对路径，并把 stdout/stderr 重定向到 `/mnt`，不要在 crontab 或 Git 中保存 SSH 密码。

## 数据与产物

重型产物优先使用 `$ACPBGRL_DATA_ROOT`；未设置时，若 `/mnt/songensheng` 可写则使用 `/mnt/songensheng/ac-pbgrl`，否则使用项目内 `.runtime`。主要目录为：

```text
labels/{train,validation}/        HDF5 future-gain shards
replay/<run>/                     分字段 mmap ring buffer
runs/<method>/seed_<n>/           config、manifest、metrics、checkpoint、evaluation
calibration/<method>/seed_<n>.json
temporal/gru/<method>/seed_<n>.pt
paper_figures/                    PDF、CSV 和 figure_manifest.json
```

标签 shard 带 teacher hash、horizon、discount、reward 定义及 map-split hash；checkpoint、replay metadata 与 JSON 均采用原子提交。更完整的公平性、指标定义和统计方向见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。

## 离线服务器

联网机器先构造 wheelhouse：

```bash
./scripts/offline/build_wheelhouse.sh
ACPBGRL_SSH_HOST=<ssh-alias> ./scripts/offline/deploy_server.sh
```

服务器从已有 `ros_conda_packed.tar.gz` 解包到 `/mnt` 下的独立环境，不修改其他项目环境：

```bash
ACPBGRL_DATA_ROOT=/mnt/songensheng/ac-pbgrl ./scripts/offline/install_server.sh
ACPBGRL_PYTHON=/mnt/songensheng/ac-pbgrl/env/bin/python ./run.sh doctor --system server_a40
```

地址、密码和私有 SSH 配置不会写入 Git。详细离线校验与同步方式见 [docs/SERVER_AND_ROS.md](docs/SERVER_AND_ROS.md)。

## ROS Noetic / ONNX

本地 `ros_noetic` 容器中：

```bash
./scripts/offline/bootstrap_ros.sh
cd /root/ros_ws/ARE
catkin_make --pkg ac_pbgrl_ros
source devel/setup.bash
roslaunch ac_pbgrl_ros ac_pbgrl.launch model_path:=/path/to/full.onnx
```

ROS 包订阅 `/projected_map`（`nav_msgs/OccupancyGrid`）和 `/state_estimation`（`nav_msgs/Odometry`），发布 `/way_point`（`geometry_msgs/PointStamped`）。Python 3.8/Noetic 使用 `onnxruntime==1.16.3`；引导脚本离线安装到项目内 `.runtime/ros_python`，不污染系统 Python。导出模型采用固定 node/candidate padding 和 mask，并生成同名 JSON 元数据。

## 验证范围

```bash
./run.sh test
./run.sh train --config full --gpus cpu --smoke
./run.sh export --config full --checkpoint /path/to/checkpoint.pt --validate --smoke
```

交付验证覆盖公式/掩码、action index 不变式、真实地图适配、label/replay、KF/EMA/GRU、动态 batch、OOM 回退、DDP 恢复、统计制图、ONNX 数值一致性和 ROS topic waypoint 闭环。正式多种子百万 transition 训练属于论文实验，不是代码 smoke test，结果不能在运行前预设。
