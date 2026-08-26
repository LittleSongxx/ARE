# 论文改进计划汇报 PPT

## 交付文件

- 主文件：`KF-Enhanced-DRL-Exploration-论文改进计划.pptx`
- 可重复生成脚本：`create_kf_plan.js`
- 生成命令：`node presentation/create_kf_plan.js`
- `KF-v2-presentation.html` 仅作早期技术备查，不代表当前 PPT 的视觉风格或论文主线

## 当前论文定位

汇报将最终方法命名为：

> **UPBG-RL: Uncertainty-aware Predictive Belief Graph Reinforcement Learning**

核心问题不是“再增加几个工程模块”，而是补上 baseline 对未知区域缺少显式长期判断这一环：

> 机器人不再只比较候选节点眼前有多少 frontier，而是判断继续向该方向探索，可能打开多少当前尚未发现的自由空间，并持续维护这份判断的可信度。

## 汇报主线

前半部分先完整解释两篇 baseline，再自然引出最终方法：

> 探索任务 → ARiADNE 的图注意力选点 → Large-scale DRL 的稳定训练与大图部署 → 已发表效果与剩余缺口 → UPBG-RL

- ARiADNE 把局部地图转换为带 utility 和访问状态的候选图，用图注意力 Actor 在可行邻居中连续选择下一视点，并用离散 SAC 学习兼顾新信息、行程代价和任务完成度的策略。
- Large-scale DRL 保留 Actor 主干，训练时让 Critic 读取完整真值图以降低价值估计噪声；部署时通过图稀疏化保留信息区域与关键通路，使小场景训练的策略能够处理大场景。
- 两篇工作共同建立了图决策、长期回报学习和大场景部署基础，但未知区域的未来收益仍主要由策略隐式推断，缺少可解释的潜力表示、可信度维护和按决策价值组织的大图结构。

## 最终优化思路

### 1. 预测候选方向的未来潜力

Baseline 的 utility 更接近“这里现在能看到多少 frontier”。新的预测任务回答“从这里继续走，后面可能打开多少新空间”。

训练阶段利用完整地图生成节点级教学答案，让 Actor 学会区分两类情况：

- 眼前 frontier 很多，但后面很快到达尽头；
- 眼前 frontier 一般，但后面连接新的房间或走廊。

模型不仅给出潜力判断，还给出这份判断的可信程度。这样策略可以在“高潜力但证据不足”和“潜力一般但判断可靠”之间作出明确取舍。

拟采用 **特权信息蒸馏（Privileged Knowledge Distillation）+ 异方差图回归（Heteroscedastic Graph Regression）+ RankNet 成对排序**：

- 冻结某一训练时刻的局部地图与候选图；
- 对每个候选方向在完整真值图上做有限视距 rollout；
- 统计该方向能够带来的新可达自由空间与 frontier 组，并按到达代价折扣，得到节点级 future-gain target；
- GT Teacher 读取完整图，Student 只读取 belief graph，通过特权蒸馏对齐潜力表征；
- 在共享 Graph Encoder 后增加双头 Potential Head，分别输出潜力均值与 log-variance；
- 以异方差 Gaussian NLL 学习数值和预测方差，以 RankNet 式成对排序学习邻居顺序，再与离散 SAC 联合训练。

### 2. 随地图更新持续维护可信度

每次新地图到来时，系统不会立即覆盖上一轮结论，也不会机械保留历史判断，而是根据当前证据的可靠程度决定如何更新：

- 连续观测一致时，逐步积累证据，减少单帧噪声导致的方向摇摆；
- 新区域明显打开且当前判断可靠时，快速提高该方向的优先级；
- frontier 消失、节点已访问或通路发生变化时，降低旧判断权重并重新估计。

这里维护的是“未来潜力及其可信度”，不是对 raw utility 做简单平滑。

拟采用 **结构事件门控的自适应卡尔曼滤波（Event-gated Adaptive Kalman Filter）**：

- 按空间位置、邻接关系与区域归属关联前后两帧中的持久节点；
- 每个节点保存 KF state、协方差 `P`、记忆年龄和结构事件标记；
- 以 Potential Head 的潜力均值作为观测 `z`，以预测方差设置观测噪声 `R`；
- 拓扑稳定时保持较小过程噪声 `Q`，一致观测会逐步降低 posterior covariance；
- 节点访问、frontier 消失、边失效或区域合并/分裂时，提高 `Q` 让滤波器快速跟随，或直接重置状态；
- posterior 潜力与协方差作为下一轮 Actor 的节点特征。

### 3. 用多尺度图同时表达远程方向与局部动作

大场景中不可能在固定计算预算下保留所有节点。最终方法按决策价值分配节点预算：

- 远处用区域级信息表示“哪个方向或区域更值得去”；
- 近处保留 frontier、路口、障碍变化和可行邻居等动作细节；
- 优先保留潜力高、虽不确定但值得确认、以及维持路径连通所必需的节点。

因此，大图压缩不再只是删点，而是有目的地保留影响长期选择和下一步执行的信息。

拟采用 **Random-Walk Graph Wavelet Transform + DiffPool/Top-K Pooling + A* 路径骨架**：

- 对图特征做 1/2/4-hop 随机游走平滑，得到低频 `LF`；用原始特征减 `LF` 得到高频 `HF`；
- 低频表示房间、长走廊和区域趋势，输入 DiffPool 形成远端区域 token；
- 高频突出 frontier、拐角与拓扑变化，在机器人附近根据 future potential、KF covariance 和 wavelet energy 做 Top-K 保留；
- 使用 A* 最短路径骨架补回连接区域 token 与局部动作图所需的节点，避免压缩后断路；
- Encoder 用 LF 构造全局注意力的 Query/Key，Value 分支保留原始特征与 HF residual，Pointer Decoder 最终仍只选择当前可行邻居。

## 训练与部署边界

训练和部署严格分开：

- 训练阶段：完整地图负责生成未来潜力教学答案，并提供区域级与局部级教师提示；
- 部署阶段：完整地图和教师全部移除，机器人只使用自己的局部地图，由训练后的 Actor 选择下一导航点。

一句话概括：完整地图是老师，不是机器人的额外传感器。

## Utility KF 是否保留

Raw Utility KF 不作为最终方法的核心组件保留。

原因是 frontier 和 utility 会随着新观测、节点访问和拓扑变化突然出现或消失。这些跳变往往代表真实事件，不是应该被普通平滑器抹掉的随机噪声。继续平滑 raw utility 容易带来响应滞后，并保留已经过时的判断。

最终方法保留的是更合理的上层思想：跨时间积累证据，并根据可信度与结构事件更新“未来潜力判断”。

具体实现上重新使用 Kalman Filter，但滤波状态已经从 raw utility 改为潜在的 future-potential belief。Potential Head 的预测方差设置观测噪声 `R`，结构事件调整过程噪声 `Q` 或触发重置，因此不会用平滑延迟掩盖真实 frontier 消失。

## 24 页结构

1. 封面：UPBG-RL
2. 探索任务的核心难点
3. 两代 baseline 与本文位置
4. ARiADNE 原理一：局部地图、候选图与连续选点闭环
5. ARiADNE 原理二：图注意力 Actor、Critic、离散 SAC 与奖励逻辑
6. Large-scale DRL 原理一：训练期真值 Critic 如何降低学习噪声
7. Large-scale DRL 原理二：图稀疏化与 small-to-large 部署
8. 两篇 baseline 的已发表结果与能力边界
9. 研究缺口：未知区域未来潜力仍缺少显式表示
10. 最终优化思路的一句话版本
11. 总体方法：预测、更新、组织、决策
12. 算法蓝图：特权蒸馏、自适应 Kalman Filter 与 Graph Wavelet
13. 改进一：异方差 Potential Head、RankNet 与 A/B 节点例子
14. 改进二：结构事件门控的 Adaptive Kalman Filter
15. 改进三：Graph Wavelet、DiffPool/Top-K 与 A* 连通骨架
16. 完全分离的训练与部署通道
17. 路口场景下的一次完整决策
18. Baseline 与最终方法对比
19. 三项论文贡献
20. 逐层消融与评价重点
21. 面向机器人行为的预期效果
22. 按科学风险排序的研究路线图
23. 论文依据与理论落点
24. 结论

## 汇报边界

- Cao 论文中的百分比是已发表 baseline 结果。
- UPBG-RL 页面描述的是 proposed research plan，不代表当前已经取得相应实验提升。
- PPT 不预填任何虚构提升数字，预期效果全部写成可被实验否定的行为假设。
- 部署阶段不增加完整地图、教师网络或额外传感器输入。

## 视觉与版式

配色继续参考 `宋恩圣-月度汇报 - 副本.pptx`：

- 白色背景与深灰正文；
- 主蓝 `#0F6FC6`；
- 辅助色使用青蓝、青绿、橙色、黄绿色和浅蓝；
- 封面使用蓝色横幅；
- 内容页使用左侧蓝色标题标记和短绿色分段；
- 流程图采用等宽、等高、同一水平线布局；
- 训练与部署使用两条独立水平通道，不使用斜向或交叉箭头；
- 方法页以中文解释和行为例子为主，不展示推导公式。

Baseline 部分复用了原月度汇报中的两张 ARiADNE 原图，并重新编排为当前浅色版式：

- `assets/ariadne_decision_graph.png`：地图、候选视点、图连接、utility 与执行路径；
- `assets/ariadne_policy_network.png`：局部地图、增强图、Encoder、Decoder 与策略输出；
- `assets/large_p3_fig.png`：Large-scale DRL 从点云、Octomap、信息图到稀疏图和策略网络的流程。
- `assets/upbg_value_aware_multiscale_graph.png`：1920×1080 的 Graph Wavelet 多尺度图整图，展示 LF/HF 分解、DiffPool/Top-K、区域 token、局部动作图与最终下一航点。

## 建议讲述节奏

- 第 1-3 页，约 3 分钟：问题与两代 baseline 的演进关系；
- 第 4-8 页，约 7 分钟：ARiADNE、Large-scale DRL 的基本原理、结果和能力边界；
- 第 9-12 页，约 5 分钟：研究缺口、总体方法与具体算法蓝图；
- 第 13-17 页，约 8 分钟：三个改进、训练边界与路口决策例子；
- 第 18-22 页，约 5 分钟：对比、贡献、实验、预期与路线图；
- 第 23-24 页，约 1 分钟：论文依据与总结。

总时长约 27 分钟。时间较紧时，可压缩第 5、8、17、23 页，把重点放在第 4、6-7、9-16、18-20 页。

## 生成与检查

```bash
node --check presentation/create_kf_plan.js
node presentation/create_kf_plan.js
unzip -t 'presentation/KF-Enhanced-DRL-Exploration-论文改进计划.pptx'
```

实验完成后，优先用真实结果替换第 20-21 页，并补充未来潜力排序、可信度校准、地图规模曲线、典型轨迹和失败案例。
