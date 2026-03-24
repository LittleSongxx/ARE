# Wavelet-Privileged Graph RL for Large-Scale Exploration

独立训练包，基于 `large-scale-DRL-exploration` 复制并改造成 Python 包 `wpg_rl`，默认关闭全部 wavelet 开关时保持 baseline 训练路径不变。

## 依赖

建议使用 `/root/miniconda3/envs/ros_conda/bin/python`。

```bash
cd /root/ros_ws/ARE/src/Wavelet-Privileged\ Graph\ RL\ for\ Large-Scale\ Exploration
pip install -r requirements.txt
```

如果地图不在项目内 `maps/`，可设置：

```bash
export WPG_RL_MAPS_DIR=/path/to/maps
```

## 训练

```bash
cd /root/ros_ws/ARE/src/Wavelet-Privileged\ Graph\ RL\ for\ Large-Scale\ Exploration
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py
```

快速 smoke：

```bash
cd /root/ros_ws/ARE/src/Wavelet-Privileged\ Graph\ RL\ for\ Large-Scale\ Exploration
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/smoke_test.py
```

## CPU/GPU 与结果保存

- collector/Ray worker 固定跑在 CPU；learner 由 `--use-gpu-global 0|1` 和 `--num-gpu N` 控制。
- 可显式控制 Ray 和 worker CPU 分配：`--ray-num-cpus`、`--ray-worker-num-cpus`、`--worker-num-threads`。
- 正常训练结果写到 `result/<run_session>/train/{model,tensorboard,gifs}` 和 `result/<run_session>/test/{gifs,eval}`。
- 训练图片和 gif 会按 `--result-bucket-episodes` 分桶写到 `result/<run_session>/train/gifs/episode_xxx/`。
- 自动测试 gif 会按 `--auto-eval-interval` 分桶写到 `result/<run_session>/test/gifs/episode_xxx/map_i/`。
- `test/eval` 会保存 `evaluation_history.json`、`evaluation_history.csv`、`fixed_eval_maps.txt`、原始明细 `raw/episode_xxx/eval_episode_XXXXX.json`，以及按指标拆分的逐图历史曲线目录，如 `explored_rate/`、`success_rate/`、`completion_steps/`、`completion_travel_dist/`。
- `--load-model 1` 会优先读取最近一次 checkpoint，并继续写回原来的 `run_session`。
- checkpoint 按 `--save-model-gap` 周期保存到 `checkpoint.pth`，同时按 `--result-bucket-episodes` 分桶写到 `train/model/episodes_xxx/`。
- 训练正常结束会额外写 `checkpoint_final.pth`；中断时会尽力写 `checkpoint_interrupted.pth`。
- `--smoke` 仍会创建 tensorboard/gif/eval 目录，但不写训练 checkpoint。
- 固定测试集默认取地图目录里按文件名排序后的前 `N` 张图，由 `--auto-eval-map-count` 控制，`fixed_eval_maps.txt` 会把 `map_i` 到真实地图文件的映射记录下来。
- corridor graph compression、corridor edge pruning、smoothness reward 默认都关闭；不显式开启时保持原版逻辑。

## 开关

- `--use-lf-attention-hf-residual 0|1`
- `--use-privileged-wavelet-distillation 0|1`
- `--wavelet-scales 1,2,4`
- `--wavelet-fuse-dim 128`
- `--wavelet-lf-qk 0|1`
- `--wavelet-distill-weight 0.1`
- `--wavelet-distill-lf-weight 1.0`
- `--wavelet-distill-hf-weight 1.0`
- `--wavelet-distill-warmup-updates 1000`
- `--wavelet-distill-ramp-updates 2000`
- `--ray-num-cpus N`
- `--ray-worker-num-cpus N`
- `--worker-num-threads N`
- `--num-gpu N`
- `--result-bucket-episodes N`
- `--auto-eval-map-count N`
- `--auto-eval-interval N`
- `--auto-eval-greedy 0|1`
- `--disable-auto-eval`
- `--enable-corridor-graph-compression 0|1`
- `--enable-corridor-edge-pruning 0|1`
- `--enable-smoothness-reward 0|1`
- `--corridor-max-width M`
- `--corridor-min-length M`
- `--smoothness-turn-penalty W`
- `--smoothness-lateral-penalty W`

## Ablation

baseline:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py \
  --use-lf-attention-hf-residual 0 \
  --use-privileged-wavelet-distillation 0
```

`+#1 distill`:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py \
  --use-lf-attention-hf-residual 0 \
  --use-privileged-wavelet-distillation 1
```

`+#2 LF attention/HF residual`:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py \
  --use-lf-attention-hf-residual 1 \
  --use-privileged-wavelet-distillation 0
```

both:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python scripts/train.py \
  --use-lf-attention-hf-residual 1 \
  --use-privileged-wavelet-distillation 1
```

## 测试

```bash
cd /root/ros_ws/ARE/src/Wavelet-Privileged\ Graph\ RL\ for\ Large-Scale\ Exploration
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-wpg-rl /root/miniconda3/envs/ros_conda/bin/python -m unittest discover -s tests -p 'test_*.py'
```
