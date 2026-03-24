from __future__ import annotations

# ============================================================
# Training-consistency parameters (must match checkpoint)
# ============================================================
CELL_SIZE = 0.4
NODE_RESOLUTION = 4.0
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

FREE = 255
OCCUPIED = 1
UNKNOWN = 127

SENSOR_RANGE = 20.0
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 3
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128

K_SIZE = 25
NODE_PADDING_SIZE = 360

# Model architecture switches (must match checkpoint)
USE_LF_ATTENTION_HF_RESIDUAL = True
WAVELET_SCALES = (1, 2, 4)
WAVELET_FUSE_DIM = 128
WAVELET_LF_QK = True
USE_PRIVILEGED_WAVELET_DISTILLATION = False

# ============================================================
# Corridor refinement (graph optimisation at inference time)
# ============================================================
ENABLE_CORRIDOR_GRAPH_COMPRESSION = True
ENABLE_CORRIDOR_EDGE_PRUNING = True
CORRIDOR_MAX_WIDTH = 1.5 * NODE_RESOLUTION
CORRIDOR_MIN_LENGTH = 2.0 * NODE_RESOLUTION

# ============================================================
# Planner parameters (tunable per scenario via launch)
# ============================================================
THR_TO_WAYPOINT = 4.0
THR_NEXT_WAYPOINT = 5.0
THR_GRAPH_HARD_UPDATE = 8.0
CLUSTER_RANGE = 10.0

AVOID_OSCILLATION = True
ENABLE_SAVE_MODE = True
ENABLE_DSTARLITE = False

# ============================================================
# Graph rarefaction (key-node sparse graph)
# ============================================================
ENABLE_GRAPH_RAREFACTION = True

# ============================================================
# Wavelet adaptive distance threshold
# ============================================================
WAVELET_ADAPTIVE_DTH = True
WAVELET_DTH_ALPHA = 1.0
WAVELET_DTH_MAX_MULT = 2.0
WAVELET_DTH_SCALE_MULTS = (1, 2, 4)
WAVELET_LOCAL_MAP_SIZE = 56.0
WAVELET_CACHE_CHANGE_RATIO_THRESH = 0.01

USE_GPU = False
USE_GPU_GLOBAL = False
NUM_GPU = 0
