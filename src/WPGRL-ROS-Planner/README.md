# WPGRL-ROS-Planner

Wavelet-Privileged Graph RL 规划器，适用于 CMU autonomous_exploration_development_environment 仿真环境。

## 目标

将 WPG_RL (Wavelet-Privileged Graph RL for Large-Scale Exploration) 训练得到的策略网络部署到 ROS 仿真环境中，进行闭环探索测试。

## 与 ARiADNE-ROS-Planner 的关系

- **ROS 骨架**（launch / topic / 可视化 / 规划循环）参考 ARiADNE-ROS-Planner (v1)
- **算法内核**（PolicyNet / Agent / NodeManager / 图处理）来自 WPG_RL 训练代码
- 除 WPG_RL 特有修改外，规划流程与 ARiADNE-ROS-Planner 保持一致，便于对比实验

## 与 WPG_RL 训练代码的关系

- 运行时仅使用 **actor (PolicyNet)**，不使用 critic / QNet / 蒸馏逻辑
- PolicyNet 含小波编码层 (low-frequency attention + high-frequency residual)
- 图处理使用 corridor_refinement（边剪枝 + 图压缩），不使用旧版图稀疏化

## 关键差异：地图值域 remap

ROS OccupancyGrid 值域与 WPG_RL 训练值域不同：

| 语义     | ROS 值 | WPG 训练值 |
|----------|--------|------------|
| free     | 0      | 255        |
| occupied | 100    | 1          |
| unknown  | -1     | 127        |

规划器在 `get_map_callback` 中通过 `remap_ros_occupancy_to_wpg()` 自动完成转换。

## 参数分类

### 训练一致性参数（必须与 checkpoint 匹配，不可随意修改）

| 参数 | 默认值 |
|------|--------|
| `NODE_INPUT_DIM` | 4 |
| `EMBEDDING_DIM` | 128 |
| `CELL_SIZE` | 0.4 |
| `NODE_RESOLUTION` | 4.0 |
| `SENSOR_RANGE` | 20.0 |
| `K_SIZE` | 25 |
| `NODE_PADDING_SIZE` | 360 |
| `USE_LF_ATTENTION_HF_RESIDUAL` | True |
| `WAVELET_SCALES` | (1, 2, 4) |
| `WAVELET_FUSE_DIM` | 128 |
| `WAVELET_LF_QK` | True |

### 场景调优参数（可通过 launch 覆盖）

| 参数 | launch 名称 | campus 默认 | indoor 默认 |
|------|-------------|-------------|-------------|
| waypoint 到达阈值 | `waypoint_threshold` | 4 | 3 |
| 重规划频率 | `replanning_frequency` | 1 | 1 |
| 走廊图压缩 | `enable_corridor_graph_compression` | true | true |
| 走廊边剪枝 | `enable_corridor_edge_pruning` | true | true |
| 贪心策略 | `greedy_action_selection` | true | true |

## 运行方法

### 1. 启动仿真环境（不需要 conda）

```bash
cd /root/ros_ws/ARE
source devel/setup.sh
roslaunch vehicle_simulator system_campus.launch
```

### 2. 启动规划器（需要 conda 环境）

```bash
conda activate ros_conda
cd /root/ros_ws/ARE
source devel/setup.sh
roslaunch wpg_rl_planner wpg_rl_planner_campus.launch
```

其他场景：

```bash
# indoor
roslaunch vehicle_simulator system_indoor.launch
roslaunch wpg_rl_planner wpg_rl_planner_indoor.launch

# forest
roslaunch vehicle_simulator system_forest.launch
roslaunch wpg_rl_planner wpg_rl_planner_forest.launch

# tunnel
roslaunch vehicle_simulator system_tunnel.launch
roslaunch wpg_rl_planner wpg_rl_planner_tunnel.launch

# garage
roslaunch vehicle_simulator system_garage.launch
roslaunch wpg_rl_planner wpg_rl_planner_garage.launch
```

## 替换 checkpoint

方法一：直接替换文件

```bash
cp /path/to/new/checkpoint.pth src/WPGRL-ROS-Planner/src/scripts/model/checkpoint.pth
```

方法二：通过 launch 参数指定

```bash
roslaunch wpg_rl_planner wpg_rl_planner_campus.launch model_path:=/path/to/checkpoint.pth
```

## 故障排查

| 现象 | 检查项 |
|------|--------|
| 模型 shape 不匹配 | 确认 checkpoint 与 PolicyNet 架构参数一致 |
| waypoint 不动 | 检查 /way_point topic 是否发布；检查地图 remap 是否正确 |
| 图为空 | 检查 start 点是否在 free 区域；NODE_RESOLUTION 是否过大 |
| 节点启动失败 | 确认在 conda 环境中启动；确认 torch 可用 |

## 目录结构

```
WPGRL-ROS-Planner/
  README.md
  src/
    CMakeLists.txt
    package.xml
    launch/                           5 个场景 launch
    rviz/WPGRL.rviz
    scripts/
      wpg_rl_planner.py              ROS 节点入口
      wpg_runtime/                   推理内核
        model.py                     PolicyNet（含小波编码）
        agent.py                     Agent（corridor_refinement + padding）
        node_manager.py              NodeManager + check_valid_node
        corridor_refinement.py       走廊图优化
        wavelet_graph.py             小波图分解
        sparse_visualization.py      稀疏图可视化
        utils.py                     地图/碰撞/前沿工具
        quads.py                     四叉树
        parameter.py                 推理参数
        runtime_utils.py             map remap / checkpoint 加载
      model/
        checkpoint.pth               训练好的模型文件
```
