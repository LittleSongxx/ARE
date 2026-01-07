# ROS Exploration Algorithms Workspace

统一的ROS自主探索算法研究工作空间，整合了多个探索规划算法的实现。

## 项目结构

### 探索规划算法

#### 1. **ARiADNE-ROS-Planner**
- **描述**: 原始ARiADNE探索规划器
- **特点**: 基于图的自主探索算法
- **来源**: [GitHub - ARiADNE-ROS-Planner](https://github.com/MichaelFYang/ARiADNE)

#### 2. **ARiADNE2-ROS-Planner** ⭐
- **描述**: ARiADNE增强版，集成强化学习策略
- **核心功能**:
  - 预训练的PyTorch RL策略网络
  - 完全参数化的launch配置
  - 动态节点分辨率调整
  - 社区检测与可达性验证
  - 增量式图更新
  - 方向性偏置选项
  - 增强碰撞检测
- **适用场景**: 复杂环境（校园、建筑物、狭窄通道）
- **来源**: [GitHub - ARiADNE2-ROS-Planner](https://github.com/MichaelFYang/ARiADNE2)

#### 3. **TARE Planner**
- **描述**: Hierarchical Framework for Autonomous Exploration
- **特点**: 基于视点的贪心局部覆盖规划
- **用途**: 性能基线对比
- **来源**: [GitHub - TARE Planner](https://github.com/caochao39/tare_planner)

### 强化学习框架

#### 4. **large-scale-DRL-exploration**
- **描述**: 大规模深度强化学习探索框架
- **功能**: RL策略训练、环境模拟、节点管理
- **组件**:
  - 图神经网络代理 (agent.py)
  - 并行训练驱动 (driver.py, worker.py)
  - 环境接口 (env.py)
  - 四叉树空间索引 (quads.py)

#### 5. **DARE**
- **描述**: Diffusion-based Autonomous Robot Exploration
- **特点**: 基于扩散模型的探索策略
- **技术栈**: PyTorch, Diffusion Policy, Transformer

### 仿真与可视化

#### 6. **autonomous_exploration_development_environment**
- **描述**: 自主探索开发环境
- **包含**:
  - vehicle_simulator: Gazebo车辆仿真
  - visualization_tools: RViz可视化与实时指标监控
  - terrain_analysis: 地形分析工具
  - local_planner: 局部路径规划
  - sensor_scan_generation: 传感器数据生成

## 快速开始

### 环境要求
- Ubuntu 20.04
- ROS Noetic
- Python 3.8
- Gazebo 11

### 编译工作空间
```bash
cd /root/ros_ws
catkin_make
source devel/setup.bash
```

### 运行ARiADNE2（推荐）

**启动仿真环境:**
```bash
roslaunch vehicle_simulator system_campus.launch
```

**启动探索规划器:**
```bash
roslaunch ariadne2 ariadne2_campus.launch
```

### 运行TARE Planner
```bash
roslaunch vehicle_simulator system_campus.launch
roslaunch tare_planner tare_planner.launch
```

## 参数配置

### ARiADNE2关键参数

**节点密度控制:**
- `node_resolution`: 节点分辨率 (默认: 2.0m)
- `enable_dynamic_resolution`: 动态分辨率开关

**探索效率:**
- `utility_range_factor`: 效用检测范围倍率 (默认: 0.5)
- `min_utility`: 最小效用阈值 (默认: 3)
- `frontier_cluster_range`: 前沿聚类范围 (默认: 20.0m)

**算法优化开关:**
- `use_directional_bias`: 方向性偏置
- `enable_incremental_community_update`: 增量社区更新
- `enable_community_reachability_check`: 社区可达性检查
- `enable_enhanced_collision_check`: 增强碰撞检测

详细参数说明见：`ARiADNE2-ROS-Planner/src/launch/ariadne2_campus.launch`

## 结果保存

### 探索指标图表
- **位置**: `~/exploration_results/`
- **格式**: `exploration_metrics_YYYYMMDD_HHMMSS.png`
- **内容**: 探索体积、移动距离、算法运行时间
- **保存方式**: 
  - 探索完成自动保存
  - 图表窗口按 `s` 键手动保存

## 实验对比

| 算法 | 策略类型 | 优势 | 劣势 |
|------|---------|------|------|
| ARiADNE2 | RL学习策略 | 全局最优、长期规划 | 保守，避免狭窄区域 |
| TARE | 贪心视点选择 | 激进、快速探索 | 局部最优、回溯频繁 |
| ARiADNE | 确定性图搜索 | 稳定可靠 | 缺乏学习能力 |

## Git管理

所有子项目已从独立仓库转换为统一workspace：

```bash
# 查看状态
git status

# 提交更改
git add .
git commit -m "描述修改内容"

# 查看历史
git log --oneline

# 推送到远程仓库 (需先配置)
git remote add origin <your-repo-url>
git push -u origin master
```

## 开发计划

- [ ] 混合策略：结合RL全局规划与贪心局部探索
- [ ] 参数自适应：根据环境特征自动调参
- [ ] 多机器人协同探索
- [ ] 真实机器人部署测试

## 参考文献

1. ARiADNE: A Reinforcement Learning Approach Using Attention-based Deep Networks for Exploration
2. TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments
3. Large-scale Deep Reinforcement Learning for Autonomous Driving

## 许可证

各子项目遵循其原始开源许可证，详见各目录的LICENSE文件。

---

**维护者**: song  
**最后更新**: 2026-01-07
