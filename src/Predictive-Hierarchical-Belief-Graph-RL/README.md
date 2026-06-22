# Predictive-Hierarchical-Belief-Graph-RL

HPBG-RL 是基于 Cao 2024 `large-scale-DRL-exploration` 训练主干改造的独立训练包，目标是通过 prediction-augmented belief、层级拓扑图、uncertainty-aware utility、privileged critic/reward 和多尺度蒸馏，稳定超过原 baseline。

Python 包名：`hpbg_rl`。

## 方法边界

- actor 部署期只使用在线观测、地图预测 / belief 特征和层级图上下文。
- ground-truth map、oracle utility、expert potential 等 privileged 信息只进入训练期 critic observation、expert reward shaping 或辅助蒸馏 target。
- actor 节点输入固定为 8 维：基础 4 维 + belief/prediction/hierarchy 派生 4 维。
- critic 节点输入固定为 11 维：actor 8 维 + privileged 3 维。
- 禁用 HPBG 相关模块时，新增特征保持 fixed-width neutral/zero 填充，避免 replay、worker 和 checkpoint 协议漂移。

## 依赖

建议使用 `/root/miniconda3/envs/ros_conda/bin/python`。

```bash
cd /root/ros_ws/ARE/src/Predictive-Hierarchical-Belief-Graph-RL
/root/miniconda3/envs/ros_conda/bin/python -m pip install -r requirements.txt
```

如果地图不在项目内 `maps/`，可设置：

```bash
export HPBG_RL_MAPS_DIR=/path/to/maps
```

## 训练

服务器启动脚本：

```bash
cd /home/user/songensheng/Predictive-Hierarchical-Belief-Graph-RL
./start_train_hpbg_rl.sh
```

`start_train_hpbg_rl.sh` 只负责 conda 环境、`CODE_DIR`、`PYTHONPATH`、地图路径和 resume/smoke 这类启动流程参数。核心训练参数默认从 `src/hpbg_rl/parameter.py` 读取，建议优先在该文件中统一调参。

可选启动覆盖：

```bash
MAPS_DIR=/path/to/maps RUN_NAME=hpbg_rl_exp ./start_train_hpbg_rl.sh
TRAIN_SMOKE=1 ./start_train_hpbg_rl.sh
RESUME_FROM=result/hpbg_rl/train/model/checkpoint_final.pth ./start_train_hpbg_rl.sh
```

直接入口：

```bash
cd /root/ros_ws/ARE/src/Predictive-Hierarchical-Belief-Graph-RL
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-hpbg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py
```

快速 smoke：

```bash
cd /root/ros_ws/ARE/src/Predictive-Hierarchical-Belief-Graph-RL
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-hpbg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/smoke_test.py
```

## 关键开关

HPBG 主线：

- `--use-hpbg 0|1`
- `--use-belief-state 0|1`
- `--use-map-prediction 0|1`
- `--use-hierarchical-graph 0|1`
- `--use-expert-reward 0|1`
- `--use-belief-distillation 0|1`
- `--hpbg-risk-weight 0.35`
- `--hpbg-belief-ema-alpha 0.35`
- `--hpbg-cluster-resolution 12.0`
- `--hpbg-cluster-edge-hops 1`
- `--hpbg-expert-reward-weight 0.25`
- `--hpbg-expert-potential-weight 1.0`
- `--hpbg-oracle-gain-weight 0.0`
- `--hpbg-belief-distill-weight 0.05`

多尺度图表征 / 蒸馏：

- `--use-lf-attention-hf-residual 0|1`
- `--use-privileged-wavelet-distillation 0|1`
- `--wavelet-scales 1,2,4`
- `--wavelet-fuse-dim 128`
- `--wavelet-lf-qk 0|1`
- `--wavelet-distill-weight 0.1`
- `--wavelet-distill-lf-weight 1.0`
- `--wavelet-distill-hf-weight 1.0`

运行资源 / 评测：

- `--ray-num-cpus N`
- `--ray-worker-num-cpus N`
- `--worker-num-threads N`
- `--num-gpu N`
- `--result-bucket-episodes N`
- `--auto-eval-map-count N`
- `--auto-eval-interval N`
- `--auto-eval-greedy 0|1`
- `--disable-auto-eval`
- `--maps-dir /path/to/maps`
- `--train-maps-dir /path/to/train_maps`
- `--val-maps-dir /path/to/val_maps`
- `--test-maps-dir /path/to/test_maps`
- `--split-manifest-path /path/to/split_manifest.json`
- `--split-seed N`
- `--val-map-count N`
- `--test-map-count N`
- `--run-final-test 0|1`

公平评估协议：

- 训练环境只消费 `train` split。
- 训练中的自动评估只消费 `val` split，用于泛化监控和 best validation checkpoint selection。
- `test` split 只用于训练结束后的最终报告；默认不会被训练过程访问。
- 如果提供 `split_manifest.json`，HPBG 和 Cao baseline 必须使用同一个 manifest。
- 如果未提供 manifest，会按 `split_seed` 从地图目录确定性生成互斥 `train/val/test` split，并把本次运行实际使用的 manifest 固化到 `result/<run_session>/protocol/split_manifest.json`。
- 公平对比必须保持同地图 split、同地图规模分布、同起点 / seeds、同 `sensor range`、同 `max episode step`、同评估频率、同 checkpoint selection protocol、同报告指标。
- 本项目不新增消融实验；只报告完整方法、baseline compatibility 和同协议公平对比。

独立评估入口：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-hpbg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/evaluate.py \
  --checkpoint result/<run_session>/train/model/checkpoint_final.pth \
  --split test
```

`evaluate.py` 会优先继承 checkpoint 中保存的 runtime config；显式 CLI 参数优先级最高。输出包含 summary JSON、per-map CSV、split/manifest hash、step budget、map metadata、exploration AUC 和 travel efficiency。

走廊稀疏图兼容实验：

- `--enable-corridor-graph-compression 0|1`
- `--enable-corridor-edge-pruning 0|1`
- `--enable-smoothness-reward 0|1`
- `--corridor-max-width M`
- `--corridor-min-length M`
- `--smoothness-turn-penalty W`
- `--smoothness-lateral-penalty W`

## Baseline 兼容模式

所有 HPBG / privileged / wavelet 增强关闭时，仍走 fixed-width 协议，但新增槽位为 neutral/zero 特征。

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-hpbg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py \
  --use-hpbg 0 \
  --use-belief-state 0 \
  --use-map-prediction 0 \
  --use-hierarchical-graph 0 \
  --use-expert-reward 0 \
  --use-belief-distillation 0 \
  --use-privileged-wavelet-distillation 0 \
  --use-lf-attention-hf-residual 0
```

## 结果保存

- collector / Ray worker 固定跑在 CPU；learner 由 `--use-gpu-global 0|1` 和 `--num-gpu N` 控制。
- 正常训练结果写到 `result/<run_session>/train/{model,tensorboard,gifs}` 和 `result/<run_session>/test/{gifs,eval}`。
- `--load-model 1` 会优先读取最近一次 checkpoint，并继续写回原来的 `run_session`。
- checkpoint 周期保存到 `checkpoint.pth`；正常结束额外写 `checkpoint_final.pth`；中断时尽力写 `checkpoint_interrupted.pth`。
- `--smoke` 会创建 tensorboard/gif/eval 目录，但不写训练 checkpoint。

## 测试

```bash
cd /root/ros_ws/ARE/src/Predictive-Hierarchical-Belief-Graph-RL
PYTHONPATH=src /root/miniconda3/envs/ros_conda/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src /root/miniconda3/envs/ros_conda/bin/python scripts/smoke_test.py
```
