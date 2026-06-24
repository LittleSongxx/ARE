# Autonomous Exploration Paper Strategy, 2026-06-24

## 0. Local Reproducibility Notes

The code was checked with the requested conda environment:

- Environment: `conda run -n rosclaw`, Python 3.12.13.
- Available core packages: `torch`, `numpy`, `skimage`, `matplotlib`, `pytest`, `fitz`.
- Missing training packages: `ray`, `tensorboard`.
- `src/large-scale-DRL-exploration/tests`: 8 tests passed in `rosclaw`.
- A two-step `Worker` episode now runs in `rosclaw` after replacing deprecated NumPy `ndarray.itemset` with direct assignment in `sensor.py`.
- Full `scripts/train.py --smoke` does not run yet because `ray` is missing.

This means the current repo is viable as a fast prototype base, but full training needs dependency preparation.

## 1. What the Local Code Actually Supports

The local `large-scale-DRL-exploration` code implements a simplified 2D graph-RL exploration loop:

- Belief update is an ideal 360-degree range scan in `sensor.py`.
- Robot action is direct transition to a neighboring graph node.
- Reward is distance penalty plus newly observed frontier count in `env.py`.
- Actor observation is `[relative x, relative y, utility, visited]`.
- Critic observation adds privileged ground-truth explored-state information.
- The open training repo does not contain the full paper's large-scale graph rarefaction module as a clean training-time component.

The local `ARiADNE2-ROS-Planner` contains a practical rarefied graph planner and many ROS-side heuristics, but its README says the full paper code is not released. It is better treated as an engineering reference than as a faithful training implementation.

## 2. Broad Literature Map

### MarmotLab / NUS Line

- ARiADNE: attention-based graph RL for autonomous exploration, ICRA 2023.
- Large-scale DRL exploration: privileged critic plus graph rarefaction, RA-L 2024.
- DARE: diffusion policy for autonomous robot exploration, ICRA 2025.
- MARVEL: multi-agent RL for constrained-FoV multi-robot exploration, ICRA 2025.
- HEADER: hierarchical graph, community detection, guideposts, privileged expert reward, arXiv 2025.
- GRATE: graph transformer plus Kalman smoothing for time-efficient exploration, ICRA 2026 listing.
- Their GitHub also shows active 2026 exploration-adjacent repos, including `MARVEL`, `Multi-Agent-Active-SLAM`, and `IR2`.

Sources:

- https://marmotlab.org/projects/exploration.html
- https://github.com/marmotlab
- https://arxiv.org/abs/2301.11575
- https://arxiv.org/abs/2403.10833
- https://arxiv.org/abs/2410.16687
- https://arxiv.org/abs/2510.15679
- https://arxiv.org/abs/2509.12863
- https://github.com/marmotlab/MARVEL

### Conventional Large-Scale Exploration

- TARE and the Science Robotics representation-granularity work are still central baselines for large-scale 3D exploration.
- FAEL and HPHS are strong fast/hierarchical frontier-based competitors.
- DSVP and GBP remain important graph/sampling baselines in the MarmotLab papers.

Sources:

- https://roboticsconference.org/2021/program/papers/018/index.html
- https://biorobotics.ri.cmu.edu/papers/paperUploads/scirobotics.adf0970.pdf
- https://github.com/caochao39/tare_planner
- https://github.com/SYSU-RoboticsLab/FAEL
- https://arxiv.org/abs/2407.10660

### Active SLAM and Uncertainty-Aware Learning

This space is already active, so a plain "uncertainty-aware graph RL explorer" is not enough by itself.

- Chen/Englot series: graph DRL under localization uncertainty, IROS 2020 and ICRA 2021.
- Active SLAM survey: frames active SLAM as planning actions to build accurate and complete maps, and highlights belief-space planning, DRL, multi-robot coordination, reproducibility, and practical deployment.
- MA-SLAM: large-scale map-aware Active SLAM with DRL, arXiv 2025.
- GLEAM: generalizable active mapping in complex 3D indoor scenes, ICCV 2025.

Sources:

- https://arxiv.org/abs/2007.12640
- https://arxiv.org/abs/2105.04758
- https://arxiv.org/abs/2207.00254
- https://arxiv.org/abs/2511.14330
- https://arxiv.org/abs/2505.20294

### Map Prediction / World-Model Exploration

- UPEN learns occupancy priors and uses model uncertainty for exploration/navigation.
- MapEx estimates probabilistic information gain from global map predictions.
- GUIDE combines global graph inference with diffusion-based exploration.
- Pipe Planner and related 2025 work push path-wise information gain with predicted maps.

Sources:

- https://arxiv.org/abs/2202.11907
- https://arxiv.org/pdf/2409.15590
- https://arxiv.org/abs/2509.19916
- https://arxiv.org/html/2503.07504v1

### Multi-Robot Exploration

- AIM-Mapping uses asymmetric privileged information for multi-robot exploration, T-RO 2025.
- MarmotLab has privileged communication learning for bandwidth-limited multi-robot exploration.
- MARVEL already covers constrained-FoV multi-robot exploration with graph attention.
- A 2025 survey covers multi-robot cooperative exploration systems, mapping, coordination, and communication constraints.

Sources:

- https://arxiv.org/abs/2404.18089
- https://arxiv.org/abs/2407.20203
- https://arxiv.org/abs/2502.20217
- https://arxiv.org/abs/2503.07278

### Safety, Dynamic Obstacles, and Traversability

This is a more promising gap because most high-efficiency learning explorers report path length/time/coverage, while safety is handled by downstream local planners or simple collision checks.

- A 2025 graph-RL safety-shield paper targets safe cluttered exploration. It is an important direct neighbor, but it is still a greedy local policy in a 2D cluttered/forest-like setup; it does not address large-scale MarmotLab-style informative graphs, privileged expert guidance, explicit cost critics, CVaR/traversability risk, dynamic risk, or ROS terrain integration.
- Risk-aware meta-level exploration from the NeBula/CoSTAR line shows the importance of environment history, traversability risk, and kinodynamic constraints for large-scale real-world exploration, but it switches between classical local/global policies rather than learning a risk-aware graph exploration policy end to end.
- Risk-aware traversability costmap work provides a strong modeling tool: learn tail risk / CVaR cost maps from uncertain terrain data, then feed those risks to the graph planner. This is highly compatible with `ARiADNE2-ROS-Planner` because the ROS map handler already subscribes to `/terrain_cloud`.
- DEP handles dynamic UAV exploration with incremental sampling and PRM, but is not a large-scale graph-RL method.
- HEADER's ROS README warns about negative obstacles and 2.5D limitations, which is a practical deployment gap.

Sources:

- https://arxiv.org/html/2504.11907v2
- https://arxiv.org/abs/2209.05580
- https://arxiv.org/abs/2107.11722
- https://arxiv.org/abs/2010.07429
- https://github.com/marmotlab/ARiADNE2-ROS-Planner

### Multi-Floor / Multi-Layer Exploration

- Multi-floor exploration is emerging and relevant because HEADER explicitly cannot handle multi-layer free space.
- A 2026 paper uses incremental reachable graphs and structural priors for multi-floor ground-robot exploration.
- This is high-impact but heavier engineering than the current 2D trainer supports.

Source:

- https://arxiv.org/abs/2605.23350

### Semantic / LLM / VLM Exploration

This direction is now crowded and fast-moving:

- LLM-MCoX for multi-robot coordinated exploration/search.
- ABot-Explorer / SG-Memo online semantic memory construction.
- VLM-guided frontier exploration.

Sources:

- https://arxiv.org/html/2509.26324v1
- https://arxiv.org/abs/2604.19034
- https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-172.html

## 3. Directions to Avoid as Main Contributions

Avoid these as the main paper thesis:

1. "Replace attention with graph transformer": GRATE already does this.
2. "Add diffusion policy": DARE and GUIDE already do this.
3. "Add hierarchical/community graph": HEADER already does this strongly.
4. "Plain Active SLAM with DRL": Chen/Englot, MA-SLAM, and GLEAM make this crowded.
5. "Plain multi-robot RL": MARVEL, AIM-Mapping, and bandwidth-limited communication learning are nearby.
6. "Only reproduce missing graph rarefaction": useful engineering, but weak novelty.
7. "Only limited-FoV heading planning": MarmotLab already lists active perception with limited-FoV sensors, and MARVEL covers constrained FoV in multi-robot exploration.

## 4. Recommended Main Route

### Working Title

Safe Uncertainty-Constrained Graph Reinforcement Learning for Large-Scale Autonomous Exploration

Short name: `SU-GRL` or `SAFE-GRAPH-EX`.

### Core Claim

Existing large-scale learning explorers optimize coverage efficiency but treat safety, map/pose uncertainty, negative obstacles, and dynamic risk as downstream or secondary issues. We propose a graph-RL planner that reasons over exploration reward and risk in the same informative graph, trained with privileged expert guidance and deployed with a lightweight safety shield.

### Why This Is Defensible

It does not directly compete on "better ARiADNE/HEADER/GRATE network architecture." Instead, it changes the problem objective:

- From: shortest path / shortest makespan to complete coverage.
- To: fastest safe coverage under uncertain perception, localization drift, dynamic obstacles, and traversability risk.

This gives a clear experimental axis where standard ARiADNE/HEADER-like methods can be strong on distance/time but weaker on collision, near-miss, negative-obstacle, uncertainty, and recovery metrics.

### Method Components

1. Risk-aware informative graph

Each node stores:

- Existing ARiADNE features: relative position, frontier utility, visited flag.
- Local map uncertainty: entropy around node, unknown-edge ratio, observed/unobserved boundary confidence.
- Traversability/risk: obstacle clearance, negative-obstacle score, slope/elevation variance when using ROS terrain maps.
- Dynamic risk: predicted occupancy over a short horizon, or a simple moving-obstacle risk field in the 2D trainer.
- Active-SLAM proxy: pose covariance or odometry-drift proxy accumulated along graph edges.

Each edge stores:

- Distance/time cost.
- Collision/unknown crossing probability.
- Clearance and dynamic risk.
- Optional expected localization degradation.

2. Privileged risk-aware expert

Training has access to ground-truth free/obstacle/dynamic maps and true risk. Expert action is generated by a weighted constrained planner:

- Primary: cover frontiers efficiently.
- Constraint: keep edge risk below threshold.
- Secondary: reduce map entropy and avoid localization degradation.

The reward can be dense:

- Expert imitation reward: negative normalized distance between selected waypoint and privileged expert waypoint.
- Constraint penalty: Lagrangian/CVaR penalty for risk violations.
- Exploration shaping: frontier/entropy gain.

3. Constrained graph RL

Start from the existing SAC-style discrete graph policy and add:

- Separate risk critic, or one critic for return and one for cost.
- Lagrangian constrained SAC or risk-sensitive action masking.
- A safety shield that filters infeasible neighbor actions during training and deployment.

4. Deployment fallback

If the learned policy proposes unsafe/stuck actions:

- Use a nearest safe frontier fallback.
- Log shield interventions as a metric, not just as hidden recovery.

## 5. Experiments

### Phase 1: Fast 2D Training/Validation in Current Repo

Implementation changes:

- Install missing `ray` and `tensorboard` in `rosclaw`.
- Add configurable sensor noise and false free/false occupied observations.
- Add synthetic dynamic obstacles to the 2D grid.
- Add node and edge risk features.
- Extend `NODE_INPUT_DIM`, `CRITIC_NODE_INPUT_DIM`, and tests.
- Add evaluation metrics for safety and uncertainty.

Baselines:

- Nearest frontier.
- Utility frontier.
- ARiADNE local code.
- Large-scale DRL local code.
- Large-scale DRL plus heuristic safety shield.
- Proposed full model.

Metrics:

- Exploration success rate.
- Coverage over time.
- Travel distance and makespan proxy.
- Collision count.
- Near-miss count / minimum clearance.
- Unsafe action proposal rate.
- Safety shield intervention rate.
- Map IoU/F1 against ground truth.
- Robustness under noise levels.

Expected publishable result:

- Similar or slightly longer paths than vanilla DRL, but significantly fewer unsafe events and better map quality under noise/dynamic risk.

### Phase 2: ROS/Gazebo Realistic Validation

Use:

- `autonomous_exploration_development_environment`.
- `ARiADNE2-ROS-Planner` integration points.
- Campus, indoor, tunnel, forest worlds.

Add:

- Dynamic pedestrians/vehicles or moving obstacles.
- Negative obstacle / pit / drop-off regions where possible.
- Traversability risk from terrain analysis.

Baselines:

- TARE, if available in workspace or restored.
- ARiADNE2 planner.
- FAEL/HPHS if feasible to run.
- Proposed model with and without safety shield.

Metrics:

- Explored volume/area over time.
- Distance, makespan, planning runtime.
- Collision and near-miss count.
- Negative-obstacle failures.
- Traversability violations.
- Recovery events.
- Final map completeness and consistency.

### Phase 3: Optional High-Impact Add-Ons

Only add these after Phase 1/2 results are strong:

- Pose covariance from SLAM backend or a proxy odometry-noise model.
- Ensemble map prediction uncertainty, MapEx-style, as node features.
- Multi-floor reachable-surface graph extension.
- Limited-FoV heading action.

## 6. Ablation Plan

Minimum ablations:

1. Vanilla ARiADNE/large-scale input.
2. Add risk features only.
3. Add safety shield only.
4. Add privileged expert reward only.
5. Add constrained risk critic.
6. Full model.

Stress tests:

- Increasing sensor noise.
- Increasing dynamic obstacle density.
- Narrow passages.
- Negative obstacle density.
- Long corridor / loop closure style layouts.
- Out-of-distribution maps.

## 7. Publication Targeting

Conservative SCI route:

- IEEE Access, Applied Sciences, Sensors, Robotics and Autonomous Systems, Intelligent Service Robotics, or similar Q2/Q3 venues depending on final results and indexing year.
- This route can accept strong simulation plus careful ablations.

RA-L / IROS route:

- Need ROS/Gazebo validation, strong baselines, and safety metrics.
- Real robot or rosbag replay would help a lot.
- Claim should focus on "safe uncertainty-constrained exploration" rather than "another graph RL architecture."

ICRA/RSS route:

- Requires either real robot deployment in risky terrain/dynamic environment, or a very clean constrained-RL formulation with strong generalization.
- Multi-floor or negative-obstacle real-world validation would increase novelty but also engineering risk.

## 8. Implementation Roadmap

Step 1:

- Keep the local NumPy compatibility fix.
- Install `ray` and `tensorboard` in `rosclaw`.
- Run `scripts/train.py --smoke` successfully.

Step 2:

- Add risk map and noisy sensor simulation in the 2D trainer.
- Add safety metrics to `evaluation.py`.
- Add tests for risk features and shield behavior.

Step 3:

- Extend observation features and retrain baseline models.
- Add risk critic or constrained action masking.
- Compare against vanilla trained model.

Step 4:

- Port the runtime policy to ROS planner.
- Run Gazebo experiments in at least indoor, forest, tunnel.

Step 5:

- Prepare paper figures: risk-aware graph visualization, shield interventions, safety-efficiency Pareto curves, noise robustness curves, trajectories.

## 9. Final Recommendation

The most publishable route is:

Safe uncertainty-constrained graph RL exploration with privileged risk-aware expert training and deployment-time safety shield.

This is stronger than simply completing MarmotLab's missing graph rarefaction because it changes the scientific question. It is also safer than plain Active SLAM DRL because the literature already has graph uncertainty RL, MA-SLAM, and GLEAM. The key is to make safety, uncertainty, and traversability first-class objectives and show a clear Pareto improvement: modest path/time cost for large reductions in unsafe behavior and map failures.

## 10. Sharpened Paper Thesis After Additional Literature Review

### Proposed Name

`C-SAFE-GE`: Constrained Safe Graph Exploration.

### Main Thesis

Existing learning-based autonomous exploration methods, including ARiADNE, large-scale DRL exploration, HEADER, DARE, MARVEL, and GRATE, primarily optimize efficiency: shorter distance, lower makespan, better scale, better graph reasoning, or better limited-FoV/multi-agent coordination. Active SLAM work handles localization uncertainty, and recent safe-shield work handles local cluttered navigation. The open paper opportunity is a constrained learning formulation that unifies:

- exploration gain,
- map/sensor uncertainty,
- traversability and negative-obstacle risk,
- dynamic obstacle risk,
- deployment-time safety fallback,
- and privileged training signals.

The target claim should not be "we are faster than HEADER/GRATE everywhere." The target claim should be:

> For large-scale exploration under uncertain and risky traversability, the proposed constrained graph policy achieves a better safety-efficiency Pareto frontier than efficiency-only graph RL and shield-only heuristics.

### Difference From Closest Work

- Versus MarmotLab graph RL: adds risk/cost state, constrained optimization, and safety metrics as first-class objectives instead of assuming collision-free informative graphs and low-noise SLAM.
- Versus graph-RL safety shield paper: moves from local greedy grid actions to large-scale informative graph planning, adds risk critics / Lagrangian or CVaR costs, privileged expert guidance, dynamic/traversability risks, and ROS validation.
- Versus NeBula/CoSTAR risk-aware meta-level exploration: learns graph decisions directly instead of switching hand-designed local/global planners.
- Versus risk-aware costmaps: uses learned/estimated traversability risk as a planning signal inside autonomous exploration rather than only as a navigation costmap.
- Versus Active SLAM GNN papers: focuses on safe coverage under terrain/dynamic risk, not only localization uncertainty and information gain.

### Minimum Novel Contributions

1. A risk-aware informative graph whose nodes/edges carry frontier gain, entropy, clearance, unknown-crossing risk, dynamic occupancy risk, and optional pose-covariance proxy.
2. A constrained graph-RL planner with both value and cost/risk critics, trained with a Lagrangian/CVaR-style objective and action masking.
3. A privileged risk-aware expert signal using ground truth maps/risk during training, avoiding purely handcrafted frontier rewards.
4. A deployment-time safety shield and fallback that is reported as a metric, not hidden as an implementation detail.
5. A safety-efficiency benchmark protocol for ARE: collisions, near misses, negative-obstacle failures, shield intervention rate, unsafe proposal rate, map IoU/F1, distance, makespan, and planning time.

### Recommended Experimental Bar

For a solid SCI Q3 submission:

- 2D trainer experiments with at least 100 held-out maps, 5 seeds, sensor noise, dynamic obstacles, and synthetic negative-obstacle/risk maps.
- Baselines: nearest frontier, utility frontier, vanilla ARiADNE/large-scale DRL, vanilla plus shield, risk-feature-only, constrained full model.
- Statistical reporting: mean, standard deviation, paired significance tests, safety-efficiency Pareto curves.

For a credible IROS/ICRA or stronger Q2 attempt:

- Add ROS/Gazebo validation in indoor, forest, tunnel, and campus worlds from the autonomous exploration development environment.
- Include terrain-derived risk from `/terrain_cloud` and at least one dynamic-obstacle scenario.
- Compare against ARiADNE2-ROS-Planner and one strong conventional planner if feasible, such as TARE/FAEL/HPHS.
- Add one small hardware or rosbag replay demonstration if available.

### Concrete Code Entry Points

- `src/large-scale-DRL-exploration/env.py`: add cost/risk returned by `step`, not only scalar reward.
- `src/large-scale-DRL-exploration/sensor.py`: add noisy observations and false-free/false-occupied perturbations.
- `src/large-scale-DRL-exploration/node_manager.py`: compute node risk, entropy, clearance, and edge risk.
- `src/large-scale-DRL-exploration/agent.py`: extend node observations beyond `[relative x, relative y, utility, visited]`.
- `src/large-scale-DRL-exploration/parameter.py`: extend `NODE_INPUT_DIM` and `CRITIC_NODE_INPUT_DIM`.
- `src/large-scale-DRL-exploration/model.py` / `worker.py`: add risk/cost critic and Lagrange multiplier logging.
- `src/large-scale-DRL-exploration/evaluation.py`: add collision, near-miss, unsafe proposal, shield intervention, risk integral, map IoU/F1.
- `src/ARiADNE2-ROS-Planner/src/src/map_handler.cpp`: expose terrain-derived traversability risk from `/terrain_cloud` to the planner.

## 11. Additional Angles With Feasibility Analysis

This section re-opens the search beyond the safety route and evaluates each angle by practical feasibility in the current ARE workspace, not only by theoretical novelty.

### Feasibility Anchors From The Current Workspace

- The fast 2D trainer has 5663 PNG maps, a simple `Env.step`, and an exploration reward based on distance and frontier reduction. This makes budget, return-to-base, noisy sensing, map prediction, and graph-feature ablations easy to prototype.
- The current graph observation is small: `NODE_INPUT_DIM = 4`, `CRITIC_NODE_INPUT_DIM = NODE_INPUT_DIM + 1`. Adding a few scalar node features is technically low-risk.
- The ROS development environment has five Gazebo worlds: `campus`, `forest`, `garage`, `indoor`, and `tunnel`.
- The CMU development environment README explicitly includes collision avoidance, terrain traversability analysis, waypoint following, and visualization tools.
- `ARiADNE2-ROS-Planner` subscribes to `/terrain_cloud`, but its README says the released module cannot handle multi-floor environments and cannot handle negative obstacles. This is both an opportunity and a warning: terrain-risk work is feasible, multi-floor work is much heavier.

### Candidate A: Mission-Budgeted / Return-to-Base Graph RL

Working title: `B-SAFE-GE`, Budgeted Safe Graph Exploration.

Core idea:

- Maximize explored area under a hard travel-time/energy/deadline budget while guaranteeing return to the start or a rendezvous point.
- Add `remaining_budget`, `shortest_return_cost`, and `budget_feasible` node/edge features.
- Add a shield that rejects actions whose outbound cost plus return cost exceeds the remaining budget.

Novelty:

- Budget-aware exploration exists, including MapExRL and dynamic-deadline exploration, but MarmotLab-style large-scale graph RL papers do not make return feasibility a first-class constraint.
- This angle pairs naturally with the previous safety route: safe exploration is incomplete if the robot cannot return.

Feasibility:

- High in the 2D trainer: `travel_dist` already exists in `env.py`; return distance can be computed by Dijkstra/A* on the known graph; budget violations are easy to log.
- Medium in ROS: return-to-base is feasible with the waypoint follower and grid map, but reliable runtime experiments require robust path planning to home.

Minimal implementation:

- Add episode budgets sampled from several difficulty ratios.
- Add node features: normalized remaining budget, return cost from node to start, action feasibility flag.
- Add reward/cost: exploration gain, travel penalty, budget violation penalty, return success bonus.
- Add shield: only allow actions with `cost(current,next) + return_cost(next) <= remaining_budget`.

Baselines:

- Vanilla large-scale DRL / ARiADNE.
- Nearest frontier with return feasibility.
- Utility frontier with return feasibility.
- Vanilla policy plus budget shield.
- Proposed constrained policy.

Metrics:

- Coverage before deadline.
- Return success rate.
- Budget violation rate.
- Distance and makespan.
- Safe-action proposal rate.
- Coverage-return Pareto curve.

Publication feasibility:

- SCI Q3: high if 2D + ROS simulation and good ablations are complete.
- Q2/IROS: possible if combined with safety/traversability and tested in multiple Gazebo worlds.
- ICRA/RSS: unlikely as a standalone unless real robot or a strong constrained-RL formulation is added.

Sources:

- https://arxiv.org/abs/2503.01548
- https://dl.acm.org/doi/10.1109/IROS51168.2021.9636199
- https://arxiv.org/abs/2603.15604

### Candidate B: Lightweight Map-Prediction Auxiliary Graph RL

Working title: `P-GE`, Predictive Graph Exploration.

Core idea:

- Train a lightweight map completion model from partial occupancy maps.
- Use predicted occupancy and uncertainty to estimate future frontier value and guide the graph policy.
- Use the prediction model as an auxiliary head or a frozen module, not as the full planner.

Novelty:

- Crowded. UPEN, MapExRL, SenseExpo, GUIDE, and active neural mapping already use occupancy priors, map prediction, or learned world models.
- The safest novelty angle is not "we predict maps", but "we use calibrated prediction uncertainty as a graph-level auxiliary signal for large-scale exploration and safety/budget constraints."

Feasibility:

- Medium-high in 2D: the workspace has enough PNG maps to synthesize many partial-map/full-map pairs.
- Medium in ROS: map prediction can be kept offline or run on the 2D occupancy grid, avoiding RGB/semantic perception.
- Low if framed as neural implicit/NeRF/3DGS mapping, because the current stack is LiDAR/grid-map oriented.

Minimal implementation:

- Generate partial observations from existing maps using the current sensor simulator.
- Train a small U-Net or ResNet encoder-decoder to predict full free/occupied/unknown probabilities.
- Add graph features: predicted gain, prediction entropy, structural corridor/room likelihood.
- Add auxiliary loss during RL or freeze the predictor and only use graph features.

Baselines:

- Vanilla graph RL.
- Frontier with predicted information gain.
- MapExRL/SenseExpo-style heuristic if implementable.
- Proposed graph RL plus calibrated prediction features.

Metrics:

- Map prediction IoU/F1/calibration error.
- Exploration distance/time.
- OOD generalization to held-out maps and map styles.
- Failure cases where prediction overconfidence hurts exploration.

Publication feasibility:

- SCI Q3: medium-high if implementation is clean and experiments are broad.
- Q2/IROS: medium only if combined with safety/budget or strong OOD generalization.
- As a standalone paper, novelty risk is high.

Sources:

- https://arxiv.org/abs/2202.11907
- https://arxiv.org/abs/2503.01548
- https://arxiv.org/html/2503.16000v2
- https://arxiv.org/abs/2410.16687

### Candidate C: Energy-Aware Exploration For Ground Robots

Working title: Energy-Constrained Graph Exploration.

Core idea:

- Replace pure distance cost with an execution-energy proxy that includes distance, turning, acceleration/smoothing, slope/traversability, and sensor/computation usage.
- Learn an exploration policy that trades coverage against energy, not just distance or makespan.

Novelty:

- Energy-aware exploration is active for UAVs, and energy-aware mobile robot navigation exists, but large-scale graph-RL exploration for ground robots rarely evaluates energy explicitly.
- This becomes stronger if paired with terrain risk: slope/roughness increases both risk and energy.

Feasibility:

- High for a first 2D proxy: add turn angle, acceleration proxy, path smoothness, and distance.
- Medium in ROS: can estimate motor/trajectory energy, but real battery/current measurement is not present in the current repo.
- Low for a high-tier claim without physical energy measurement.

Minimal implementation:

- Add heading state and turn cost to the 2D trainer.
- Add smoothed waypoint execution cost.
- In ROS, estimate energy from velocity/acceleration commands or terrain slope proxy.

Baselines:

- Distance-minimizing graph RL.
- GRATE-like smoothing/Kalman postprocessing if available.
- Frontier planner with energy-aware target selection.

Metrics:

- Energy proxy or measured energy.
- Coverage per joule.
- Distance/time.
- Smoothness/turn count.
- Map quality.

Publication feasibility:

- SCI Q3: medium-high with careful energy model and simulations.
- Q2/IROS: medium if ROS and real measurement are added.
- Standalone novelty is weaker than safety/budget because EAAE already targets exploration energy, albeit for UAVs.

Sources:

- https://arxiv.org/abs/2603.15604
- https://cdn.aaai.org/ocs/18463/18463-79403-1-PB.pdf

### Candidate D: Dynamic-Obstacle-Aware Graph Exploration

Core idea:

- Treat moving obstacles as a time-varying risk field on graph edges.
- Learn to select frontiers that avoid likely future conflicts and reduce replanning churn.

Novelty:

- Dynamic exploration and dynamic navigation already exist, but graph-RL autonomous exploration under dynamic risk is still less developed.
- This is best as a component of the safety route, not a standalone main contribution.

Feasibility:

- High in 2D: synthetic moving obstacles are easy to add.
- Medium in ROS: Gazebo moving actors or scripted dynamic obstacles are feasible but need stable launch setup.
- Medium risk: dynamic-obstacle experiments can become planner/local-controller debugging rather than exploration research.

Minimal implementation:

- Add moving obstacle states and predicted occupancy over a short horizon.
- Add edge dynamic-risk features and dynamic collision metrics.
- Evaluate under obstacle density and speed sweeps.

Publication feasibility:

- SCI Q3: medium if combined with safety metrics.
- Q2/IROS: medium-low as standalone; medium if integrated into `C-SAFE-GE`.

Sources:

- https://arxiv.org/abs/2010.07429
- https://arxiv.org/html/2504.16734v2

### Candidate E: Safety / Generalization Benchmark For ARE

Working title: `ARE-SafeBench`.

Core idea:

- Build a benchmark protocol on top of the current 2D maps and five Gazebo worlds, focusing on safety, uncertainty, dynamic obstacles, negative obstacles, return-to-base, and reproducibility.

Novelty:

- Explore-Bench and GLEAM-Bench already exist, so a benchmark-only paper must offer a clearly different axis.
- The missing axis is safety-constrained large-scale exploration: collisions, near-misses, shield intervention, budget violation, return success, negative-obstacle failure, and map-quality degradation under noise.

Feasibility:

- Very high for 2D.
- Medium for Gazebo.
- Good as a companion contribution to a method paper; risky as the only contribution.

Minimal implementation:

- Define scenario generators for noise, dynamic obstacles, risk maps, and deadlines.
- Standardize metrics and logs.
- Release scripts to run baselines and generate Pareto plots.

Publication feasibility:

- SCI Q3: medium-high if polished and public.
- Q2/IROS: medium only with strong baselines and ROS/real validation.

Sources:

- https://arxiv.org/abs/2202.11931
- https://arxiv.org/abs/2505.20294

### Candidate F: Multi-Robot Communication / Coordination

Core idea:

- Extend graph RL to decentralized multi-robot exploration under bandwidth limits, intermittent communication, or heterogeneous robots.

Novelty:

- Strong field, but very crowded and directly adjacent to MarmotLab's own later work: MARVEL, bandwidth-limited privileged communication, AIM-Mapping, and active multi-robot SLAM.

Feasibility:

- Low-medium in this workspace: the local trainer is single-robot; multi-agent rollout, replay, credit assignment, map merging, and collision avoidance must be built or imported.
- Higher if starting from an external MARVEL/multi-agent repo, but that becomes a different codebase.

Publication feasibility:

- High upside, high engineering risk.
- Not recommended as the next paper from the current ARE code unless a multi-agent repo becomes the primary base.

Sources:

- https://arxiv.org/html/2404.18089v3
- https://arxiv.org/abs/2407.20203
- https://github.com/marmotlab/MARVEL
- https://arxiv.org/html/2502.20217v1

### Candidate G: Multi-Floor / Multi-Layer Exploration

Core idea:

- Replace 2.5D planning with a reachable-surface graph that handles stairs, ramps, overpasses, and stacked free space.

Novelty:

- High practical value, and `ARiADNE2-ROS-Planner` explicitly cannot handle multi-floor environments.
- However, recent multi-floor exploration with structural priors already makes the direction visible.

Feasibility:

- Low in the current trainer: the 2D grid representation cannot express layered free space.
- Low-medium in ROS: terrain analysis and point clouds exist, but a full 3D/layered graph, reachability model, and evaluation maps are needed.
- This is a thesis-scale direction, not a quick paper extension.

Publication feasibility:

- IROS/ICRA potential if implemented well.
- Not recommended as the immediate next route unless substantial engineering time is available.

Sources:

- https://arxiv.org/abs/2605.23350
- https://github.com/marmotlab/ARiADNE2-ROS-Planner

### Candidate H: Semantic / VLM / LLM-Guided Exploration

Core idea:

- Use semantic labels, open-vocabulary perception, VLM affordances, or LLM planning to decide where to explore.

Novelty:

- Very crowded and moving fast.
- Better suited for object search or semantic mapping than pure LiDAR exploration.

Feasibility:

- Low in this workspace: current fast trainer has occupancy maps only; ROS stack has camera topics, but no semantic labels, simulator labels, object tasks, or VLM pipeline.
- Would require a new dataset/task definition and heavy perception infrastructure.

Publication feasibility:

- High-tier possible in general, but not the best match for the current codebase.
- Not recommended as the main route.

Sources:

- https://arxiv.org/html/2509.26324v1
- https://arxiv.org/abs/2604.19034
- https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-172.html

### Candidate I: Neural / 3DGS Active Mapping

Core idea:

- Use NeRF, Gaussian splatting, or neural maps, then plan views to reduce reconstruction uncertainty.

Novelty:

- Active neural mapping is strong and current, but it is a different problem setting from MarmotLab's LiDAR grid/graph exploration.

Feasibility:

- Low in this workspace: requires camera rendering, pose/appearance reconstruction, neural map training, and different metrics.
- Current code can at most inspire topological planning, not directly support this route.

Publication feasibility:

- Not recommended unless the project is intentionally redirected away from the current ARE graph-RL code.

Sources:

- https://arxiv.org/abs/2409.20276
- https://arxiv.org/html/2412.17769v2

### Updated Feasibility Ranking

1. Best main route: `C-SAFE-GE`, constrained safe graph exploration with risk-aware training and shielded deployment.
2. Best new alternative / extension: `B-SAFE-GE`, budgeted return-to-base safe graph exploration.
3. Best auxiliary module: lightweight map prediction with calibrated uncertainty, used as graph features rather than as the full contribution.
4. Best benchmark companion: `ARE-SafeBench`, a safety/budget/dynamic-risk evaluation protocol.
5. Defer unless resources are large: multi-robot communication, multi-floor exploration, semantic/VLM exploration, neural/3DGS active mapping.

The strongest revised paper route is therefore:

> Safe Budgeted Graph Reinforcement Learning for Large-Scale Autonomous Exploration under Traversability and Return-to-Base Constraints.

This route keeps the original safety contribution, adds a very practical mission-level constraint, and remains implementable in the current codebase.
