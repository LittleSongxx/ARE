# Safe Budgeted Graph RL

This package is a standalone research prototype for safe, budgeted autonomous exploration.
It intentionally does not depend on Ray or TensorBoard so that smoke runs work in the
`rosclaw` conda environment.

## Smoke commands

```bash
conda run -n rosclaw python src/safe-budgeted-graph-rl/scripts/train_sbge.py --smoke --seed 7
conda run -n rosclaw python src/safe-budgeted-graph-rl/scripts/run_baselines.py --smoke --seed 7
conda run -n rosclaw python -m pytest src/safe-budgeted-graph-rl/tests -q
```
