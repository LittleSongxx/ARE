# Safe Budgeted Graph RL

This package is a standalone research prototype for safe, budgeted autonomous exploration.
It intentionally does not depend on Ray or TensorBoard so that smoke runs work in the
`rosclaw` conda environment.

## Reproducible experiment commands

```bash
python src/safe-budgeted-graph-rl/scripts/make_splits.py --seed 0

conda run -n rosclaw python src/safe-budgeted-graph-rl/scripts/train_sbge.py --smoke --seed 7
conda run -n rosclaw python src/safe-budgeted-graph-rl/scripts/run_baselines.py --smoke --seed 7
conda run -n rosclaw python -m pytest src/safe-budgeted-graph-rl/tests -q

python src/safe-budgeted-graph-rl/scripts/run_experiment.py --smoke --seeds 7 8
python src/safe-budgeted-graph-rl/scripts/run_experiment.py --seeds 0 1 2 --train-episodes 50 --eval-episodes 10
```

Inside the `my_ros_noetic:v3` Docker container used in this workspace:

```bash
docker exec ros_noetic bash -lc 'cd /root/ros_ws/ARE && /root/miniconda3/envs/ros_conda_py311/bin/python src/safe-budgeted-graph-rl/scripts/make_splits.py --seed 0'
docker exec ros_noetic bash -lc 'cd /root/ros_ws/ARE && /root/miniconda3/envs/ros_conda_py311/bin/python -m pytest src/safe-budgeted-graph-rl/tests -q'
docker exec ros_noetic bash -lc 'cd /root/ros_ws/ARE && /root/miniconda3/envs/ros_conda_py311/bin/python src/safe-budgeted-graph-rl/scripts/run_experiment.py --smoke --seeds 7 8 --output-dir /tmp/sbge_experiment_smoke'
```

The experiment runner writes per-seed train/eval/baseline outputs plus `all_results.csv`,
`aggregate.csv`, and `aggregate_summary.json`.
