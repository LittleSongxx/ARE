# 离线训练服务器与 ROS Noetic 部署

## 1. 服务器目录

- 源码：`/home/user/songensheng/AC-PBGRL`（可由环境变量覆盖）；
- 独立 Python 环境：`/mnt/songensheng/ac-pbgrl/env`；
- 标签、replay、checkpoint、结果：`/mnt/songensheng/ac-pbgrl`；
- 根分区只保存源码和小型清单。

部署脚本只读取 `ACPBGRL_SSH_HOST`、`ACPBGRL_SSH_USER` 和 `ACPBGRL_REMOTE_DIR`。推荐在本机 `~/.ssh/config` 设置别名；不要把 IP、密码、私钥或 token 写入项目。

## 2. 离线依赖

在联网的 x86_64 主机运行：

```bash
./scripts/offline/build_wheelhouse.sh
(cd .wheelhouse && sha256sum -c SHA256SUMS --quiet)
ACPBGRL_SSH_HOST=<alias> ./scripts/offline/deploy_server.sh
```

训练环境锁定 Python 3.8、PyTorch 2.4.1/CUDA 12.1、Ray 2.10、NumPy 1.24.3。基础压缩环境负责提供 CUDA PyTorch；wheelhouse 提供锁定的 Ray、科学计算、评测、测试和 ONNX 依赖。部署脚本会在服务器端再次核验 wheelhouse 的 SHA256；安装后必须运行 `doctor`，确认报告中的版本与锁文件一致。

结果同步：

```bash
ACPBGRL_SSH_HOST=<alias> ./scripts/offline/sync_results.sh
```

wheelhouse、模型、replay、HDF5、ONNX 和运行缓存均在 `.gitignore` 中，不应提交。

## 3. 共享 A40 资源

```bash
./run.sh doctor --system server_a40
./run.sh supervise --config full --system server_a40 \
  --gpu-policy prefer-idle --min-gpus 1 --max-gpus 4
```

若两张卡空闲而另两张正在工作，`prefer-idle` 只选择空闲卡。DDP 进程组运行期间不热增减 GPU；资源变化触发 checkpoint 退出，下一会话可变为 4/2/1 卡。查看：

```bash
tail -f /mnt/songensheng/ac-pbgrl/supervisor/<run>/events.jsonl
```

OOM 只降低下一会话 micro-batch。所有终止信号都限定为 supervisor 自己创建的 process group。

## 4. ROS Noetic

在挂载该工作区的 `ros_noetic` 容器内运行：

```bash
cd /root/ros_ws/ARE/src/AC-PBGRL
./scripts/offline/bootstrap_ros.sh
cd /root/ros_ws/ARE
catkin_make --pkg ac_pbgrl_ros
source devel/setup.bash
```

`bootstrap_ros.sh` 使用 `.wheelhouse/ros-noetic-py38`，将 NumPy 1.24.3 和 ONNX Runtime 1.16.3 安装到 `.runtime/ros_python`。这兼容没有 `python3.8-venv` 的精简镜像。

导出并验证：

```bash
./run.sh export --config full --checkpoint /path/to/latest.pt \
  --output /path/to/full.onnx --validate
roslaunch ac_pbgrl_ros ac_pbgrl.launch model_path:=/path/to/full.onnx
```

同名 JSON 包含固定 shape、校准温度、KF 参数和 topic contract。ROS 图构建器保留当前节点与全部候选 waypoint，以空间哈希构边并为远端区域构建 context token；ONNX 输出 slot 再映射到 `candidate_xy`，不会发布虚拟 region token。

## 5. 最小验收

1. `catkin_make --pkg ac_pbgrl_ros` 成功；
2. PyTorch—ONNX `max_abs_error < 1e-4`；
3. 构造 OccupancyGrid 后，所有有效节点有 self-loop、候选坐标有限；
4. 发布 `/projected_map` 与 `/state_estimation` 后在超时内收到 `/way_point`；
5. reset service 清空 stable-ID temporal records。
