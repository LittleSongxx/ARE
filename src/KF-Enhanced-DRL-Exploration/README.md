# KF-Enhanced-DRL-Exploration

基于 [large-scale-DRL-exploration](https://github.com/marmotlab/large-scale-DRL-exploration) (Cao et al., RAL 2024) 的增强版本。

融合了 11 篇强化学习论文（聚焦卡尔曼滤波 + RL）的核心思想，补全了原代码中缺失的图稀疏化算法，并引入 KF 增强的训练管线。

## 相较于原始代码的改进

| 模块 | 改进内容 | 来源论文 |
|------|---------|---------|
| `graph_rarefaction.py` | 图稀疏化算法：锚点识别 + 链收缩 + 距离裁剪 | Cao RAL 2024 (原论文，官方未开源此部分) |
| `kalman_filter.py` | 通用 KF 模块：标量 KF / 奖励基线 / Q值追踪 / 位置去噪 | KRPO, LKTD, KARNet, Sim-to-Real |
| `agent.py` | 图稀疏化集成 + KF 位置去噪 | Cao RAL 2024 + arXiv:2303.07243 |
| `driver.py` | KF 动态优势估计 + TensorBoard 指标 | KRPO (arXiv:2505.07527) |
| `node_manager.py` | 节点 KF 状态跟踪 utility 演化 | KARNet (arXiv:2305.14644) |
| `env.py` | Domain randomization 噪声注入 | arXiv:2303.07243 |

## 新增文件

```
KF-Enhanced-DRL-Exploration/
├── graph_rarefaction.py          # 图稀疏化核心算法
├── kalman_filter.py              # KF 模块集合
├── PAPER_ANALYSIS.md             # 11 篇论文的详细分析
├── tests/
│   ├── test_graph_rarefaction.py # 19 个稀疏化测试
│   └── test_kalman_filter.py     # 15 个 KF 测试
└── (其余文件继承自 large-scale-DRL-exploration)
```

## 运行测试

```bash
cd src/KF-Enhanced-DRL-Exploration
python3 -m pytest tests/test_graph_rarefaction.py tests/test_kalman_filter.py -v
```

## 使用方式

### 图稀疏化

图稀疏化已自动集成到 `agent.py` 的 `update_observation()` 中。当图节点数超过 `NODE_PADDING_SIZE` 时自动触发：

```python
from graph_rarefaction import graph_rarefaction

selected_indices, sparse_adj = graph_rarefaction(
    coords, utility, adj, current_index,
    max_nodes=NODE_PADDING_SIZE - 1,
)
```

### KF 动态优势估计

已自动集成到 `driver.py` 的 `train_step()` 中，KF 指标会写入 TensorBoard：
- `KF/Reward Baseline`: 动态跟踪的奖励基线
- `KF/Reward Uncertainty`: 基线不确定性
- `KF/Advantage`: KF 归一化的优势值

### KF 位置去噪

在创建 Agent 时启用：

```python
agent = Agent(policy_net, device="cpu", enable_position_kf=True)
```

### Domain Randomization

在创建 Env 时传入噪声参数：

```python
env = Env(
    episode_index,
    position_noise_std=0.1,   # 位置高斯噪声标准差
    sensor_noise_prob=0.01,   # 信念图单元翻转概率
)
```
