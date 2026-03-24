# ARiADNE_Wavelet

`ARiADNE_Wavelet` 是基于 ARiADNE 的工程化训练包，包含：
- 可选 `Wavelet History Branch`（历史序列建模，low/high 频分支融合）。
- 可选 `Wavelet Utility Auxiliary Loss`（future-related 监督，支持 `td_bootstrap` 与 `n_step_return`）。
- Curr 风格结果目录、checkpoint、终端训练日志，以及 CPU 采样 + GPU 学习调度。

## 目录与产物

训练输出统一写入：
- `result/<run_session>/train/model`
- `result/<run_session>/train/tensorboard`
- `result/<run_session>/train/gifs`
- `result/<run_session>/train/monitor`
- `result/<run_session>/test/eval`
- `result/<run_session>/test/gifs`

checkpoint 文件：
- `checkpoint.pth`
- `checkpoint_final.pth`
- `checkpoint_interrupted.pth`

## 依赖

建议在 `ros_conda` 环境下运行（需包含 `torch`, `ray`, `matplotlib`, `imageio`, `tensorboard`）。

地图路径优先级：
1. `ARIADNE_MAPS_DIR`
2. `ARiADNE_Wavelet/maps`
3. `src/maps`
4. 工作区 `maps/`
5. `src/ARiADNE_curr/maps`

## 训练命令

进入 `src` 目录后执行：

```bash
python -m ARiADNE_Wavelet.scripts.train_wavelet
```

快速 smoke：

```bash
python -m ARiADNE_Wavelet.scripts.train_wavelet --smoke
```

如果地图不在默认位置：

```bash
ARIADNE_MAPS_DIR=/root/ros_ws/ARE/maps python -m ARiADNE_Wavelet.scripts.train_wavelet --smoke
```

## 消融示例

baseline（全关）：

```bash
python -m ARiADNE_Wavelet.scripts.train_wavelet \
  --enable-wavelet-history 0 \
  --enable-wavelet-utility-loss 0
```

history only：

```bash
python -m ARiADNE_Wavelet.scripts.train_wavelet \
  --enable-wavelet-history 1 \
  --history-encoder-mode wavelet_split \
  --enable-wavelet-utility-loss 0
```

utility only：

```bash
python -m ARiADNE_Wavelet.scripts.train_wavelet \
  --enable-wavelet-history 0 \
  --enable-wavelet-utility-loss 1 \
  --utility-target-type n_step_return \
  --utility-loss-mode spatial2d
```

history + utility：

```bash
python -m ARiADNE_Wavelet.scripts.train_wavelet \
  --enable-wavelet-history 1 \
  --history-encoder-mode wavelet_split \
  --enable-wavelet-utility-loss 1 \
  --utility-target-type n_step_return \
  --utility-loss-mode spatial2d
```

## Utility 监督模式

- `--utility-target-type td_bootstrap`：基于 detached TD 目标。
- `--utility-target-type n_step_return`：Worker 端计算 n-step reward proxy（由 `--utility-target-horizon` 控制）。
- `--utility-loss-mode basic`：仅 masked regression。
- `--utility-loss-mode spatial2d`：局部 2D patch + Haar wavelet loss。

## History 相关参数

- `--history-len`
- `--history-input-dim`
- `--history-feature-set`（逗号分隔）
- `--history-wavelet-levels`
- `--history-embed-dim`
- `--history-encoder-mode`：`mlp_only` / `wavelet_shared` / `wavelet_split`

## 测试

```bash
PYTHONPATH=/root/ros_ws/ARE/src /root/miniconda3/envs/ros_conda/bin/python -m unittest discover -s /root/ros_ws/ARE/src/ARiADNE_Wavelet/tests -p 'test_*.py'
```

新增 `test_smoke_train_step.py` 可在无 Ray、无真实地图环境下验证最小前后向闭环。
