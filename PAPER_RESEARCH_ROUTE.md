# 面向大规模自主探索的论文研究路线

> **文档性质**：研究决策备忘录，不是当前代码已经完成的功能清单。
> **评估日期**：2026-08-26
> **基线**：Cao 等，ARiADNE（2023）与 Deep Reinforcement Learning-based Large-scale Robot Exploration（2024）；代码目录 `src/large-scale-DRL-exploration`。
> **工作名称**：Action-Conditioned Future-Potential Belief Graph RL（简称 **AC-PBGRL**，名称可在正式投稿前调整）。
> **权威性**：本文件替代项目根目录此前的方案分析稿；其他项目 README 只描述各自代码快照，若与本文的论文判断冲突，以本文为准。

## 0. 一页结论

当前最值得投入的论文路线不是把卡尔曼滤波、图小波、DiffPool、A* 和若干训练技巧并排组合，而是围绕一个清晰问题构建单一方法：

> **在部分可观测的大规模探索中，如何在不改变部署端观测和可执行动作集合的前提下，学习每个候选动作的未来探索潜力及其可信度，并维护这个潜力的短期时序 belief？**

建议的论文主线为：

1. 保留 Cao 2024 的 belief Actor、privileged GT Critic、离散 SAC 和当前邻居 Pointer 动作接口。
2. 用训练期可获得的 GT map 对每个当前候选动作做固定策略的有限视距 rollout，产生与 baseline reward 对齐的 action-conditioned future-gain 标签。
3. 在 belief graph 上增加 **Action-Conditioned Potential Head**，预测每个候选动作的潜力均值与条件不确定性。
4. 若引入时序 belief，先把完整动作分数分解为“具有稳定 ID 的节点/区域潜力”和“当前 edge/path 代价”。对预测方差做独立校准，再用 **event-gated adaptive Kalman filter** 跟踪前者；一次性候选 edge 没有重复观测，不能为了保留 KF 而强行滤波。
5. 将 posterior belief 真正注入 Pointer logits 或候选特征。只训练一个不参与决策的辅助头不能构成策略改进。
6. 只有在大图实验表明确有尺度瓶颈时，再加入 **action-preserving multiscale graph diffusion context**。远端 region token 只能提供上下文，不能替换当前可执行动作节点。

这条路线的严谨定位是：

- **理论合理**：future-gain 是一个有明确条件的 rollout return 估计问题，异方差回归、校准和短期状态估计都有成熟理论基础。
- **工程可接入**：训练期已有 Actor belief graph、Critic GT graph 和候选坐标对齐基础设施。
- **创新边界清楚**：创新集中在 action-conditioned potential distribution、privileged supervision 与 temporal belief 的组合，而不是重新声称 GT Critic、A* 或图稀疏化是原创。
- **需要诚实表述**：当前 `KF-Enhanced-DRL-Exploration_v2` 尚未实现这套方法；它目前主要是图稀疏化、reward-scale KF、可选位置 KF 和噪声增强的实验骨架。

---

## 1. 研究对象与 baseline 的真实边界

### 1.1 任务形式化

自主探索可以建模为部分可观测决策过程（POMDP）：

\[
(s_t,o_t,a_t,r_t),\qquad b_t \approx p(s_t\mid o_{0:t},a_{0:t-1}) .
\]

机器人只能看到 belief map 和由其构造的局部/全局图，而训练模拟器可以额外提供完整地图。当前时刻的动作不是任意连续位姿，而是从机器人节点的 collision-free 邻居中选择一个 waypoint：

\[
G_t=(V_t,E_t),\quad u_t\in V_t,\quad A_t=\mathcal N_{G_t}(u_t).
\]

部署策略必须满足：

\[
\pi_\theta(a_t\mid G_t^{belief},u_t),\quad a_t\in A_t .
\]

任何层级化或池化设计都不能悄悄把这个动作契约改成“选择一个不可执行的虚拟区域”。

### 1.2 ARiADNE 2023 的基础

ARiADNE 使用 viewpoint graph、带图掩码的 attention encoder、decoder 和 pointer network，并通过 SAC 学习探索策略。其论文已经明确指出网络会**隐式预测不同区域的 potential gains**，从而在当前 frontier 与长期探索之间做折中。

因此，论文不能声称 baseline 完全没有长期价值建模。更准确的研究问题是：

> 将隐式的长期潜力变成可监督、可校准、可随时间更新的 action-conditioned belief，是否能改善部分可观测和大尺度场景中的动作排序？

### 1.3 Cao 2024 large-scale baseline 的基础

2024 版本在 ARiADNE 上加入了训练期 privileged GT Critic，用完整地图降低部分可观测下的价值估计噪声；论文还描述了面向大地图的 graph rarefaction、A* 路径和 line-of-sight 简化流程。

这意味着下面几项不能单独包装为新贡献：

- 训练期使用 GT Critic；
- A* 路径骨架；
- 为满足节点预算而做的普通图稀疏化；
- “大图需要全局/局部层级表示”这一动机本身。

它们可以作为基线能力、实现组件或对照实验，但论文的新增机制必须有独立的因果假设和消融证据。

还要区分论文描述与本地代码状态：本地 `src/large-scale-DRL-exploration` 目前能看到 A*/Dijkstra 辅助函数，但没有完整接入论文所述的 rarefaction 调用链。若用 v2 的稀疏化实现做实验，必须明确它是对论文流程的复现/工程补全，而不是把“补齐缺失代码”写成新的学习算法贡献。

### 1.4 本地 baseline 的代码契约

真实代码中有几个会直接限制算法设计的事实：

- `agent.py` 的 Actor 节点输入目前只有相对 `x/y`、utility、visited 四维特征。
- `model.py` 的 Actor 和 Critic 都使用 6 层带图掩码的 attention encoder；Pointer 最终只从 `current_edge` 中取候选节点。
- `agent.py` 的 `select_next_waypoint()` 将 action slot 映射回原始节点坐标，并由环境执行该 waypoint。
- `parameter.py` 固定 `NODE_PADDING_SIZE=360`、`K_SIZE=25`；节点数和候选数有硬上限。
- `ground_truth_node_manager.py` 的 GT 图会包含 belief 图没有的未探索节点，节点数量和顺序不天然相同。
- `worker.py` 只保证当前节点、当前候选坐标与 GT observation 对齐，不能把两张图的全部 hidden tensor 按数组下标直接蒸馏。
- `driver.py` 的 SAC batch 字段是固定的；新增辅助标签必须显式加入 buffer，并控制 loss 权重、warm-up 和 ramp-up。

相关代码位置：

- [baseline agent.py](src/large-scale-DRL-exploration/agent.py)
- [baseline model.py](src/large-scale-DRL-exploration/model.py)
- [baseline worker.py](src/large-scale-DRL-exploration/worker.py)
- [baseline ground_truth_node_manager.py](src/large-scale-DRL-exploration/ground_truth_node_manager.py)
- [baseline env.py](src/large-scale-DRL-exploration/env.py)

---

## 2. 论文的中心假设与贡献边界

### 2.1 中心假设

当前 belief graph 对某些候选动作存在 observation aliasing：两个看起来相似的 frontier 可能通向完全不同的未知空间。GT map 在训练期可以揭示这种差异，但部署期不可用。若用 GT rollout 产生动作条件标签，并让 belief-only Actor 学习这些标签，则可以把部分“隐式长期推理”变成一个可诊断的预测任务。

论文要检验的不是“额外模块越多越好”，而是以下三个假设：

**H1：可预测性。** 在固定 rollout 规则下，belief graph 的候选特征能够预测未来探索收益；显式 potential head 的候选排序优于当前 utility 排序和普通辅助回归。

**H2：不确定性有用。** 经过校准的预测方差能够识别候选动作的标签噪声/条件不确定性，并改善策略在地图突变和 OOD 场景中的选择稳定性。

**H3：短期 belief 有用。** 对具有跨图快照稳定身份的节点/区域潜力做时序递推，可以减少方向抖动和无效回溯，同时在 frontier 消失、edge 失效等结构变化时及时重置，不牺牲探索完成效率。

H3 有一个先决条件：同一 latent state 必须在多个 planning step 中被重复观测。当前 baseline 每选择一个邻居便立即移动到该邻居，同一有向 `current_edge` 通常只存在一个决策周期。如果只能得到一次观测，KF 没有信息融合意义，应改用直接校准预测或显式的 recurrent graph memory。

多尺度图上下文是条件性假设：只有当节点规模实验显示 flat encoder 的远端信息确实不足时，才将它纳入完整模型；否则不把它强行塞进主方法。

### 2.2 建议的论文贡献表述

正式论文最多主张以下三点：

1. **Action-conditioned future-potential supervision**：提出与探索 reward 对齐的 GT rollout 标签，并在 belief graph 上学习候选动作的潜力分布。
2. **Calibrated temporal region-potential belief**：将预测方差校准后用于具有稳定节点/区域 ID 的事件门控状态估计，再与当前 edge/path 特征组合为动作分数，处理连续观测与图结构突变的矛盾。
3. **Action-preserving scale extension（可选）**：在不改变原始动作支持集的前提下，以多尺度图扩散和远端区域上下文缓解固定节点预算带来的尺度问题。

Reward baseline、位置去噪、domain randomization、A* 和普通链收缩应放在“训练稳定性/系统实现/对照组件”中，而不是与上述三点平行地声称为学术贡献。

---

## 3. 主方法：动作条件的未来潜力分布

### 3.1 Future-gain 标签必须先定义清楚

对当前状态、候选动作 $a$ 和 GT 地图，执行固定的 rollout policy $\pi_{roll}$ 共 $H$ 步，定义：

\[
y_t(a)=
\mathbb E_{\tau\sim\pi_{roll}}
\left[
\sum_{k=1}^{H}\gamma_g^{k-1}
\left(\alpha\,\Delta F_{t+k}-\lambda\,\Delta L_{t+k}\right)
\middle|
b_t,a_t=a,G_t^{GT}
\right].
\]

其中：

- $\Delta F$ 应优先使用 baseline reward 中“新观测 frontier”的定义；
- $\Delta L$ 是真实执行路径或 A* 路径的代价；
- 同一传感器覆盖区域只能计数一次，避免 rollout 中重复覆盖虚增收益；
- $H$ 和 $\gamma_g$ 是标签定义的一部分，不能在实验后随意改变；
- $\pi_{roll}$ 应是冻结的 baseline、专家或明确版本的行为策略，不应与正在更新的 Actor 形成循环目标；
- GT 只用于训练期生成 $y_t(a)$，部署端不能读取 GT map。

当前环境 reward 在 `env.py` 中主要由 frontier 变化和 waypoint 距离构成，探索完成时还有终止奖励。因此“新增自由空间面积”可以作为另一个有意义的辅助标签，但必须明确称为 auxiliary information-gain target，不能直接等同于 SAC return。

#### 与 privileged Critic 的冗余风险

baseline 的 Actor 已经通过 SAC policy loss 使用 privileged GT Critic 对当前动作给出的 Q 值。Future-gain head 如果只是换一种方式回归同一个 return，可能与现有监督高度重复，最终没有策略增益。

因此必须加入以下对照：

- baseline privileged Critic；
- 将 GT Critic 的 Q 值直接蒸馏给 belief Actor；
- 固定 GT rollout 的 future-gain supervision；
- future-gain distribution + calibration；
- future-gain distribution + temporal belief。

只有当显式标签在候选排序、校准、OOD 或最终策略上提供 GT Q distillation 没有提供的收益，才能主张它是独立贡献。它的主要差异应是“可解释的有限 horizon 信息收益分解和可信度”，而不是笼统地再次使用 privileged information。

### 3.2 标签生成的两种实现方式

**优先方案：离线/异步标签生成。** 采样 worker 保存候选 action、GT map 状态和路径信息，使用冻结 rollout worker 批量生成标签。这样不会把昂贵的 GT rollout 放进每一次梯度更新，也避免标签随当前 Actor 漂移。

**备选方案：在线短 rollout。** 只在训练早期或低频率更新标签缓存，并使用固定随机种子/固定 rollout policy。在线方案实现简单，但要监控模拟器开销和标签分布漂移。

### 3.3 Potential Head 的结构

Potential Head 必须以候选动作而不是孤立节点为单位。对当前节点 $u_t$ 和候选 $v\in A_t$，建议使用：

\[
h_t^v=f_\theta\left([
h_t^{u_t},h_t^v,e_t^{u_t,v},c_t]
\right),
\]

其中：

- $h_t^{u_t},h_t^v$ 是共享 graph encoder 后的表示；
- $e_t^{u_t,v}$ 至少包含欧氏距离/图距离、是否直接 collision-free、路径长度等已有图信息；
- $c_t$ 是当前图的全局 context；
- 直接 action head 输出 $\mu_t^{act}(u,v)$ 和 $s_t^{act}(u,v)=\log\sigma_t^2(u,v)$；采用分解时另行输出 $\mu_t^{reg}(v)$。

只从裸节点 embedding 预测 potential 会丢失“从哪里出发”和“走哪条边”的条件信息，不足以支撑 action-conditioned claim。

如果后续要引入 KF，建议显式分解：

\[
\mu_t^{act}(u,v)=
\mu_t^{reg}(v)
+\rho_\theta(h_t^u,h_t^v,e_t^{u,v},c_t)
-\lambda d_t(u,v).
\]

其中：

- $\mu_t^{reg}(v)$ 表示从持久节点/区域 $v$ 出发可获得的内在未来信息潜力，可以跨多个图快照观测；
- $\rho_\theta$ 表示当前起点、edge、局部拓扑带来的 interaction residual；
- $d_t(u,v)$ 是当前路径代价；
- KF 只跟踪 $\mu_t^{reg}(v)$，不跟踪随当前节点立即变化的完整 action score。

如果不使用时序模块，直接预测 $\mu_t^{act}(u,v)$ 是更简单的实现。若要使用区域 KF，则 Potential Head 需要同时为仍存在于图中的稳定节点/region token 产生 region-potential observation，而不能只为一次性的 `current_edge` 输出值。

### 3.4 异方差 Gaussian NLL

基本损失为：

\[
\mathcal L_{NLL}=
\frac12\left[
\exp(-s_t(a))(y_t(a)-\mu_t(a))^2+s_t(a)
\right].
\]

实现时需要：

- 对 $s$ 做上下界裁剪，避免方差塌缩或爆炸；
- 对极端 rollout 标签做稳健处理，比较 Smooth L1、Gaussian 和 Student-t 回归；
- 使用 state/action mask，忽略 padding candidate；
- 采用低权重、warm-up 或 ramp-up，防止辅助任务早期压过 SAC；
- 记录标签的零比例、长尾程度和不同地图尺寸的分布。

Gaussian 方差的正确语义是条件残差/观测不确定性（aleatoric-style predictive uncertainty），不是自动得到的 epistemic uncertainty。若论文需要声称 OOD model uncertainty，应另外使用 deep ensemble、MC dropout 或其他模型不确定性估计，并进行独立验证。

由于相同 belief observation 可能对应多个不同 GT 布局，future-gain 还可能是多峰分布。单高斯只能用均值和方差近似这种 observation aliasing；若 held-out 标签呈明显多峰，应比较 quantile regression、mixture density 或 ensemble predictive distribution，而不是继续强化不成立的高斯假设。

### 3.5 RankNet 只作为排序辅助项

对于同一状态的候选对 $(i,j)$，若标签差异足够大，可使用：

\[
\mathcal L_{rank}=-\log\sigma(\mu_i-\mu_j).
\]

建议的 pair 规则：

- 仅保留 $y_i-y_j>\delta$ 或 $y_j-y_i>\delta$ 的 pair；
- 近似相等的标签记为 tie，不强迫网络排序；
- 每个状态随机采样有限 pair，避免 $O(|A_t|^2)$ 爆炸；
- 只用 $\mu$ 排序，不让预测方差直接改写监督标签；
- 与 SAC Q 的排序结果报告 Kendall/Spearman 相关性，证明 RankNet 是否带来独立收益。

RankNet 来源：[Burges et al., Learning to rank using gradient descent](https://doi.org/10.1145/1102351.1102363)。它是合适的训练工具，但本身不是探索领域的新颖性来源。

### 3.6 Potential 必须进入实际策略

辅助头若不影响 Pointer，部署行为不会改变。最小可行的候选融合形式为：

\[
\ell_t(a)=
\ell_t^{base}(a)+
\beta_\mu\,\tilde\mu_t(a)-
\beta_\sigma\,\phi(\tilde\sigma_t(a)),
\qquad a\in A_t,
\]

也可以将 $[\mu,\sigma,e]$ 拼接到候选 embedding 后再送入 Pointer。无论采用哪种方式，都必须保持：

- logits 只对当前真实 `current_edge` 计算；
- padding action 永远被 mask；
- action index 与环境 waypoint 坐标不变；
- Critic 的 GT 输入只用于训练，不把 GT state-specific feature 传给部署 Actor。

### 3.7 Teacher–Student 的正确使用方式

第一版不建议蒸馏整个 GT encoder hidden state。更稳妥的结构是：

```text
GT map / GT graph（训练期）
    -> 固定 rollout
    -> 每个当前候选动作的 scalar future-gain

belief graph（训练期和部署期都可用）
    -> Student graph encoder
    -> Potential Head
    -> Pointer policy
```

如果后续确实需要 hidden distillation，应按世界坐标或候选 edge key 对齐，并只蒸馏当前候选集合或区域级统计量；不能按 GT 图和 belief 图的数组下标硬对齐。所有 GT feature 到 Student 的辅助监督都应 stop-gradient。

特权 Critic 的经验收益有理论解释，但不是无条件保证。部分可观测环境下，asymmetric actor-critic 的偏差与 filter stability、观测别名等条件有关；论文应把 GT 作为训练期标签来源，而不是声称部署策略拥有完整状态。

---

## 4. 时序模块：事件门控的区域潜力 belief filter

### 4.1 为什么不继续滤 raw utility

当前节点 utility 是可见 frontier 数量，具有明显的跳变：frontier 被发现时突然增加，被观测或节点访问后可直接归零。这不符合“固定节点上的平滑随机游走”假设。

因此，Utility KF 应继续默认关闭。它最多作为反事实消融，不能作为论文主线。

### 4.2 先证明滤波状态可以跨时间持久

当前 baseline 每个决策周期都会执行一个邻居 waypoint，然后把该邻居变成新的 current node。因此完整的有向动作 $(u,v)$ 通常只被打分一次。若机器人移动后就把滤波器重置，那么 KF 退化成单次测量；若不重置，又会把不同条件下的动作分数错误地当成同一状态。

\[
g_t^v=
\text{持久节点/区域 }v\text{ 在当前 belief map 下的内在未来潜力}.
\]

推荐让滤波对象成为 $g_t^v$，再在动作层使用：

\[
q_t(u,v)=g_t^v+\rho_t(u,v)-\lambda d_t(u,v).
\]

这样，region potential 可以随地图逐步变化并被重复观测，而当前起点和路径代价仍由 action-conditioned residual 负责。状态 key 至少应包含：

- 目标节点世界坐标、稳定 node ID 或稳定 region ID；
- 当前图版本或区域版本号。

图稀疏化、region split/merge 或节点删除时，需要显式的 old-to-new identity mapping。若无法建立稳定 ID 和重复观测，时序模块应从主方法中删除；不能改成对随机变化的 tensor index 做 KF。

### 4.3 基本递推

局部随机游走模型可写为：

\[
x_t=x_{t-1}+w_t,\qquad z_t=x_t+v_t,
\]

\[
P_t^- = P_{t-1}+Q_t,
\]

\[
K_t=\frac{P_t^-}{P_t^-+R_t},\qquad
m_t=m_t^-+K_t(z_t-m_t^-),
\]

\[
P_t=(1-K_t)P_t^- .
\]

在下面的递推式中，$x_t^v$ 代表 $g_t^v$；$z_t^v$ 是 Potential Head 对稳定节点/区域输出的 $\mu_t^{reg}(v)$，而不是 raw utility，也不是完整的一次性 edge action score。

### 4.4 预测方差到测量噪声的映射

不能未经校准地令 $R_t=\sigma_t^2$。建议在 held-out 数据上先做方差校准，再使用：

\[
R_t=\operatorname{clip}(c\,\hat\sigma_t^2+\epsilon,
R_{min},R_{max}).
\]

需要检查：

- NLL 与 RMSE 是否同时改善；
- 50%/80%/95% 预测区间覆盖率；
- reliability diagram 或 regression calibration curve；
- innovation 的均值、方差和 NIS（normalized innovation squared）；
- 不同地图尺寸、传感器噪声和 OOD split 上的校准是否失效。

如果方差被用作测量噪声，则方差越大意味着当前预测越不可信，Kalman gain 应越小；如果又把同一方差当成 exploration bonus，则语义相反，必须在模型中明确区分这两个用途。

### 4.5 事件门控是 hybrid estimator，不是经典 KF 最优性定理

建议的事件规则：

- **稳定阶段**：图结构和 frontier 变化小，使用较小 $Q_t$；
- **创新异常**：innovation/NIS 连续超阈值时增大 $Q_t$，但要有上下界和 cooldown；
- **frontier 消失、节点访问**：重置或退役对应 node/region state；
- **edge 失效**：删除该 edge 的 action residual/history，但只有在目标区域本身失效时才删除 region state；
- **region split/merge、稀疏图重建**：增加图版本号，禁止旧 token 的 posterior 直接复用；
- **候选重新出现**：按坐标/edge history 选择保守先验，并记录 cold-start 标志。

这实际上是 event-gated adaptive Kalman filter 或 switching/hybrid state estimator。经典 Kalman 最优性只在给定的线性、高斯、模型正确等条件下成立；事件门控和人工噪声倍率属于有依据的工程自适应机制，不能写成保持经典最优性。

### 4.6 必须与简单时序基线比较

至少比较：

- 无时序滤波；
- EMA；
- 固定 $Q/R$ 的 KF；
- innovation-adaptive KF；
- event-gated adaptive KF；
- 可选 GRU/KalmanNet 风格学习滤波器。

若 adaptive KF 不能在相同计算预算下减少方向切换、回溯或结构事件后的恢复时间，就没有理由把它保留为论文贡献。

参考：[Kalman, 1960](https://doi.org/10.1115/1.3662552)、[Mehra, 1970](https://doi.org/10.1109/TAC.1970.1099422)、[KalmanNet](https://arxiv.org/abs/2107.10043)。机器人探索中也已有将 KF 用于 waypoint smoothing 的工作 [GRATE](https://arxiv.org/abs/2509.12863)，因此必须明确本路线的 KF 跟踪的是持久区域 potential belief，而不是运动学 waypoint 后处理。

---

## 5. 大尺度扩展：动作保持的多尺度图上下文

这一部分是可选增强，不应在没有尺度实验证据时与主线绑定。

### 5.1 先区分 graph diffusion 与 canonical graph wavelet

令 $A$ 为图邻接矩阵，$D$ 为度矩阵，随机游走算子为：

\[
P=D^{-1}A .
\]

多尺度扩散特征可以写为：

\[
H^{(1)}=PH,\qquad H^{(2)}=P^2H,\qquad H^{(4)}=P^4H,
\]

并用 $H-H^{(k)}$ 表示相应的局部残差。这个定义可以称为 multiscale graph diffusion 或 diffusion-wavelet-inspired features。

严格的谱图小波通常基于图 Laplacian 的谱核：

\[
T_g^s=g(s\mathcal L),
\qquad
\mathcal L=I-D^{-1/2}AD^{-1/2}.
\]

如果正式论文使用“Graph Wavelet Transform”，应给出 Laplacian、尺度、滤波核和近似方法；否则不要把简单的 $P^kH$ 直接等同于 canonical spectral graph wavelet。

参考：[Hammond et al., Wavelets on Graphs](https://arxiv.org/abs/0912.3848)、[Diffusion Wavelets](https://doi.org/10.1016/j.acha.2006.04.004)。

### 5.2 baseline 已经有多层图 attention

baseline 的 6 层 masked attention 已能传播若干 hop 的信息。因此增加 1/2/4-hop diffusion 不会自动产生新能力，必须验证：

- 远端区域的 action ranking 是否改善；
- frontier、junction 和 topology change 的对比度是否提高；
- 是否发生 oversmoothing；
- 放在 encoder 前、encoder 后还是并行分支最有效；
- 节点规模增加时是否降低 planning latency 或显存。

建议第一版只把扩散特征作为节点特征增强：

\[
H_{aug}=\operatorname{MLP}([H,PH,P^2H,P^4H,H-PH]),
\]

不要一开始就同时改写整个 encoder、decoder 和动作图。

### 5.3 Pooling 的正确角色

标准 DiffPool 学习 assignment matrix：

\[
S=\operatorname{softmax}(\operatorname{GNN}_{pool}(X,A)),
\quad
X'=S^TZ,\quad A'=S^TAS.
\]

如果实现只是按坐标 bucket 后做 mean pooling，应称为 coordinate-based region pooling 或 region tokenization，不要直接称为 DiffPool。

论文中建议采用双分支：

```text
原始局部动作图
    -> 保留当前节点、当前候选、frontier 和关键拓扑节点
    -> local encoder

远端非动作节点
    -> multiscale diffusion
    -> region tokens / global context

global context
    -> cross-attention 注入局部节点
    -> Pointer 仍只在原始 current_edge 上选动作
```

远端 token 是上下文，不是未经展开的动作。

参考：[DiffPool](https://arxiv.org/abs/1806.08804)、[Graph U-Nets](https://arxiv.org/abs/1905.05178)、[Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)。

### 5.4 Top-K 与 A* 的论文边界

Top-K 保留规则必须是 action-preserving。至少强制保留：

- 机器人当前节点；
- 当前所有可行动作节点；
- frontier/information 节点；
- junction、dead-end 等拓扑锚点；
- 从机器人到重要远端区域的 collision-free skeleton 节点。

可研究的新增评分是：

\[
score(v)=
\alpha\,\tilde\mu(v)
+\beta\,\psi(\tilde\sigma(v))
+\gamma\,E_{diff}(v)
-\lambda\,d_{path}(v),
\]

但必须把 $\psi$ 的语义写清楚：风险惩罚、不确定性探索奖励和测量可信度不是同一件事。

A* 只应作为连通性/路径骨架工具。Cao 2024 已经使用 A* 与 line-of-sight 稀疏化，故新的论点应是“potential/uncertainty-aware retention”，而不是“加入 A*”。任何稀疏图产生的新边都必须经过 collision-free 和 line-of-sight 检查，不能用简单 BFS 连接替代真实可行路径。

### 5.5 为什么不能直接删除 action nodes

直接 Top-K 或 pooling 删除动作节点会破坏：

- `current_edge` 的原始索引；
- Actor/Critic 当前候选对齐；
- `select_next_waypoint()` 的坐标映射；
- 环境执行的碰撞约束；
- 真实 path length 与 reward 归属。

如果未来确实要把远端 region 作为高层动作，必须同时设计高层 token 到具体 waypoint/path 的展开器、失败回退和奖励分摊。这将是另一个层级规划问题，不应在第一版论文中隐含引入。

---

## 6. 与当前 v2 代码的差距

### 6.1 已有功能的准确定位

`src/KF-Enhanced-DRL-Exploration_v2` 当前主要包含：

- `graph_rarefaction.py`：anchor 识别、degree-2 chain contraction、distance-aware pruning；
- `kalman_filter.py`：标量 KF、reward baseline tracker、Q tracker、PositionKF；
- `driver.py`：RewardBaselineKF 用于 reward target 的尺度归一化；
- `agent.py`：可选 PositionKF 和图稀疏化；
- 参数中的 sensor/position noise 与 domain randomization；
- Utility KF legacy 开关，但默认关闭。

这些功能可以作为实验基础和工程对照，但不能在论文中写成已经完成的 AC-PBGRL。

此外，v2 的稀疏图实现必须先通过独立的几何可行性测试：压缩或修补后的每条边都要满足 collision-free/line-of-sight，compact graph 的连通性检查必须以真实机器人节点为根，而不能默认使用数组下标 0。否则图在张量层面连通，不代表 waypoint 在环境中可执行。

### 6.2 尚未实现的主线模块

当前代码中没有真正实现：

- GT finite-horizon future-gain rollout；
- action-conditioned Potential Head；
- heteroscedastic NLL；
- tie-aware RankNet；
- calibrated predictive variance；
- persistent node/region potential KF；
- Graph Wavelet 或正式 graph diffusion feature branch；
- learned DiffPool 或 action-preserving region-context hierarchy。

因此当前 v2 的运行结果不能直接作为上述论文方法的实验结果。

### 6.3 Utility KF 的最终决策

Utility KF 不纳入主论文路线，原因是：

- utility 由 frontier 集合变化决定，跳变明显；
- 节点访问后会被硬置零，不是平稳随机游走；
- 现有预测只影响稀疏化评分，不直接进入 policy logits；
- 每节点维护滤波器增加复杂度，但缺乏明确的独立收益；
- 与“潜力 belief”主线概念重叠且容易混淆。

如需完整性，可在附录做 `raw utility`、EMA、Utility KF 的失败/无收益消融；不要把它恢复为核心模块。

### 6.4 工程基线与论文贡献的分层

| 组件 | 论文中的正确角色 |
|---|---|
| Graph rarefaction / chain contraction | baseline 复现或大图工程支撑；除非提出新的潜力感知保留准则，否则不是核心原创 |
| RewardBaselineKF | 训练稳定性控制变量；需与 running std、EMA 等比较，不宜作为主贡献 |
| PositionKF | 传感器/部署鲁棒性控制变量；与 potential KF 研究问题不同 |
| Domain randomization | sim-to-real 实验设置 |
| Utility KF | 默认关闭，作为反事实消融 |
| Potential Head + calibrated temporal belief | 论文主贡献候选 |
| Multiscale diffusion context | 只有在尺度实验成立时作为第二阶段贡献 |

---

## 7. 最小可发表实现路线

### Phase 0：复现和环境固定

1. 固定 Cao 2024 baseline 的地图 split、传感器、动作空间、训练预算和随机种子。
2. 解决运行环境问题：安装兼容的 PyTorch；将 `sensor.py` 中已被 NumPy 2.0 移除的 `ndarray.itemset` 改成兼容写法，单独记录为环境修复。
3. 先获得 baseline 的训练曲线、完成距离、makespan、planning time 和图节点数量。
4. 验证当前 `current_edge`、坐标映射和 GT/Actor 候选对齐，没有稀疏化引入的非法边。

### Phase 1：离线 future-gain 标签

1. 保存每个状态的 belief graph、GT map、当前节点和当前候选 edge。
2. 使用冻结的 baseline/专家 rollout 生成 $y_t(a)$。
3. 检查标签的尺度、零膨胀、长尾和不同 horizon 的稳定性。
4. 先做一个不改 policy 的离线预测实验，确认 future-gain 确实可以从 belief graph 预测。

### Phase 2：Potential Head 接入 SAC

1. 在共享 graph encoder 后构造 candidate-conditioned head。
2. 先只加入 Smooth L1 或 Gaussian NLL。
3. 让 $\mu$ 通过候选融合进入 Pointer；保留原始 SAC loss。
4. 用低权重和 warm-up，防止辅助监督破坏已有 privileged critic 学习。
5. 使用坐标/edge key 对候选标签对齐，不对齐全部 GT hidden tensor。

### Phase 3：不确定性校准与时序 belief

1. 在独立 held-out 地图上校准 $\sigma^2$。
2. 与 EMA、固定 KF、innovation-adaptive KF 做公平比较。
3. 以 `(stable node/region ID, graph version)` 为状态 key，并验证同一 key 在多个 planning step 中确实得到重复观测。
4. 为访问、frontier 消失、edge invalidation、region split/merge 建立 reset/retire 语义。
5. 将 posterior mean/variance 注入 Pointer，并测量方向切换和事件响应时间。

### Phase 4：大尺度上下文（条件启用）

1. 先加入可复现的 $P^kH$ diffusion features，不直接声称 canonical wavelet。
2. 只有在节点预算成为瓶颈时，增加 global region tokens。
3. region tokens 仅通过 cross-attention 提供 context；原始 action nodes 和 `current_edge` 保留。
4. 对比不加 hierarchy、普通 pooling、potential-aware action-preserving pooling。
5. 记录额外计算量、显存、图构造时间和非法路径率。

### Phase 5：论文决策

- 若 Phase 2 已能稳定改善候选排序和策略指标，主论文只写 Potential + calibration + temporal belief。
- 若 Phase 3 对结构变化和 OOD 有独立收益，再把 adaptive filter 提升为第二贡献；如果没有稳定 latent identity 或没有重复观测，直接删除该模块。
- 若 Phase 4 只降低节点数但没有策略/系统收益，把它作为工程附录；不要为凑“第三个创新”保留复杂 hierarchy。
- 若 future-gain 在 held-out belief 上不可预测，停止堆叠模块，重新检查标签定义、观测充分性和任务是否需要预测地图而非预测 utility。

---

## 8. 实验和消融协议

### 8.1 必须的主线消融

1. Cao 2024 baseline。
2. `+ Potential Head`，仅标量 future-gain 回归。
3. `+ heteroscedastic NLL`，加入预测方差。
4. `+ RankNet`，验证排序辅助项的独立收益。
5. `+ calibrated adaptive KF`。
6. `+ multiscale diffusion features`。
7. `+ action-preserving global context`。
8. 完整模型。

每一步都要保持相同动作空间、地图 split、传感器、网络规模预算（或明确报告增加的参数/计算量）。

### 8.2 策略与系统指标

- 完成探索所需 travel distance；
- makespan / wall-clock time；
- explored rate 或 explored volume；
- episode return；
- planning latency；
- graph node count、显存和 CPU/GPU 占用；
- direction switches、backtracking、repeated edges；
- 非法 edge、碰撞或 path execution failure；
- small-to-large transfer；
- indoor-to-outdoor 或跨地图分布 OOD；
- sensor/pose noise 下的鲁棒性。

### 8.3 预测和不确定性指标

- future-gain MAE/RMSE；
- Gaussian NLL；
- candidate ranking 的 Kendall tau、Spearman rho、top-1 accuracy；
- 预测区间 coverage、interval score 和 calibration curve；
- innovation/NIS 统计；
- 结构事件后的恢复步数；
- 方差与实际误差的相关性；
- label horizon、rollout policy 和 map size 的敏感性。

### 8.4 统计要求

- 至少多个随机种子，并报告均值、标准差或置信区间；
- 地图按环境实例划分，不能让同一布局的轻微变体同时出现在训练和测试；
- 预先固定超参数搜索预算；
- 分开报告平均性能和失败案例；
- 同时报告“效果”和“代价”，避免用更大的网络或更多 rollout 计算冒充算法机制收益。

---

## 9. 理论边界与审稿风险

### 9.1 能够主张的理论结果

在固定 rollout policy、给定 GT 状态和有界 reward 的条件下，有限 horizon rollout return 是该 rollout policy 的条件期望的 Monte Carlo 估计；Potential Head 的最优平方回归解是 belief observation 下的条件均值。这足以支撑“训练期特权监督学习未来潜力”的合理性。

如果进一步证明 belief filter 的误差界，需要额外假设：

- node/region identity 在一段时间内稳定；
- target dynamics 在局部时间窗内近似随机游走或线性模型；
- 预测方差经过有效校准；
- event detector 能在有限延迟内识别结构变化。

这些假设在真实探索环境中不总是成立，因此更稳妥的是给出条件性分析和实验验证，而不是声称全局最优。

### 9.2 不能主张的结论

- 不能说 baseline 没有 long-term potential；它已经隐式学习了 potential gain。
- 不能说 GT Critic 或 Teacher–Student 本身是新的探索算法。
- 不能把单一异方差 head 的方差叫作 epistemic uncertainty。
- 不能说 event-gated KF 保持经典 KF 的最优性。
- 不能把简单随机游走多阶传播无条件称为谱图小波变换。
- 没有 learned assignment matrix 时不能声称实现了标准 DiffPool。
- 不能把 A* skeleton 作为独立原创贡献。
- 不能把当前 v2 的 reward KF/PositionKF 结果当成 future-potential belief 方法的结果。

### 9.3 主要风险与处理

| 风险 | 可能原因 | 处理 |
|---|---|---|
| future-gain 学不出来 | belief 观测不足、标签 horizon 太长、rollout policy 漂移 | 缩短 horizon、固定 teacher、增加 edge/path 特征、做可预测性诊断 |
| Potential Head 有预测收益但策略无提升 | 只做辅助监督或 loss 权重不当 | 将 posterior 进入 Pointer，做 policy-level ablation |
| 方差不校准 | Gaussian 假设不匹配、标签重尾 | temperature/variance scaling、Student-t、ensemble，对 coverage 负责 |
| KF 没有可滤的连续状态 | 同一 `current_edge` 只出现一次、tensor index 不稳定 | 改为持久 region potential；若无法建立 stable ID 则删除 KF |
| KF 延迟真实 frontier 事件 | Q 太小、region identity 误复用 | event reset、TTL、innovation gate、hard retire |
| hierarchy 破坏动作执行 | 删除候选或伪造稀疏边 | action-preserving mask、坐标映射、collision/LOS 验证 |
| 创新性被认为是组件堆叠 | 贡献横跨多个成熟模块 | 缩短主线，围绕一个 action-conditioned belief 问题组织消融 |
| 与近期工作重叠 | hierarchical graph、privileged reward、KF smoothing 已出现 | 明确区分 potential belief 与 waypoint smoothing，并做强相关工作对比 |

---

## 10. 相关工作与创新定位

### Baseline 与任务

- [ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration (Cao et al., 2023)](https://arxiv.org/abs/2301.11575)
- [Deep Reinforcement Learning-based Large-scale Robot Exploration (Cao et al., 2024)](https://arxiv.org/abs/2403.10833)
- [large-scale-DRL-exploration official repository](https://github.com/marmotlab/large-scale-DRL-exploration)

### 特权信息、蒸馏与部分可观测 RL

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Unifying distillation and privileged information](https://arxiv.org/abs/1511.03643)
- [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542)
- [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674)
- [Provable Partially Observable Reinforcement Learning with Privileged Information](https://arxiv.org/abs/2412.00985)
- [A Theoretical Justification for Asymmetric Actor-Critic Algorithms](https://arxiv.org/abs/2501.19116)

这些工作支持训练期使用特权信息，但也提醒：在 POMDP 中，特权 Critic 或 expert distillation 的收益依赖观测别名、filter stability 和模型假设，不能写成无条件无偏。

### 未来信息增益与预测地图

- [SEER: Safe Efficient Exploration for Aerial Robots using Learning to Predict Information Gain](https://arxiv.org/abs/2209.11034)
- [Robotic Exploration of Unknown 2D Environment Using a Frontier-based Automatic-Differentiable Information Gain Measure](https://arxiv.org/abs/2011.05323)
- [MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions](https://arxiv.org/abs/2409.15590)
- [PIPE Planner: Pathwise Information Gain with Map Predictions](https://arxiv.org/abs/2503.07504)

这些工作表明 future information gain / predicted map 不是空白问题；本路线的差异应落在 action-conditioned graph potential、privileged training target、uncertainty calibration 和 temporal belief 的组合接口上。

### 不确定性与滤波

- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552)
- [On the identification of variances and adaptive Kalman filtering](https://doi.org/10.1109/TAC.1970.1099422)
- [KalmanNet](https://arxiv.org/abs/2107.10043)
- [GRATE](https://arxiv.org/abs/2509.12863)

GRATE 已将 KF 用于 waypoint smoothing，因此本研究必须强调 KF 的状态是探索潜力 belief，而非运动控制轨迹。

### 图小波、扩散与层级图

- [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/abs/0912.3848)
- [Diffusion Wavelets](https://doi.org/10.1016/j.acha.2006.04.004)
- [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)
- [Graph U-Nets](https://arxiv.org/abs/1905.05178)
- [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)
- [Demystifying Oversmoothing in Attention-Based Graph Neural Networks](https://arxiv.org/abs/2305.16102)
- [HEADER: Hierarchical Robot Exploration via Attention-Based Deep Reinforcement Learning with Expert-Guided Reward](https://arxiv.org/abs/2510.15679)

近期已经出现 hierarchical graph、global/local reasoning 和 privileged reward，故不能只用“层级图 + 特权信息”作为新颖性论据。

---

## 11. 最终决策规则

在继续扩大模型前，按以下规则做研究决策：

1. **先证明标签有用**：Potential Head 在 held-out belief graph 上要比 raw utility 和简单回归基线更好地预测/排序 future-gain。
2. **再证明不确定性可信**：方差必须有 coverage/calibration 证据，不能只展示一条 loss 曲线。
3. **再证明滤波有因果收益**：Adaptive KF 必须减少抖动、回溯或事件恢复时间，并且不显著损害探索完成指标。
4. **最后才证明尺度收益**：hierarchy 只有在同等动作接口和计算预算下改善大图效率时才进入主论文。
5. **任何模块没有独立消融收益就降级**：放入工程附录或删除，不为凑模块数量保留。

论文最终应该让读者能回答一句话：

> **机器人为什么在当前信息不完整时选择这个邻居？因为模型从训练期特权地图学到了该动作的未来探索潜力，部署时把持久区域潜力、当前路径代价与局部交互组合起来，并只在存在稳定时序身份时使用经过校准的历史证据。**

这比“加入了若干成熟算法模块”更集中、更可验证，也更符合当前 baseline 的真实接口。

---

## 12. 当前验证状态

本备忘录的算法判断基于：

- 本地 baseline 和 v2 源码静态核查；
- Cao 2023/2024 原论文；
- privileged information、future information gain、异方差不确定性、Kalman/adaptive filtering、graph wavelet、DiffPool 和近期探索工作的公开资料对照。

尚未完成完整训练复现。当前环境存在两个独立的运行阻塞：缺少 PyTorch，以及 baseline `sensor.py` 使用了 NumPy 2.0 移除的 `ndarray.itemset`。它们是环境兼容问题，不应被解释为上述算法路线不可行。
