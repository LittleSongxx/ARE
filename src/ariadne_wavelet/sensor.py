import math

import numpy as np
import numba as nb


@nb.njit(cache=True)
def _collision_check_jit(x0, y0, x1, y1, ground_truth, robot_belief):
    """Bresenham ray-cast: update *robot_belief* in-place along the ray."""
    x0 = round(x0)
    y0 = round(y0)
    x1 = round(x1)
    y1 = round(y1)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    h, w = ground_truth.shape[0], ground_truth.shape[1]
    collision_flag = 0
    max_collision = 10

    while 0 <= x < w and 0 <= y < h:
        k = ground_truth[y, x]
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief[y, x] = k

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx


@nb.njit(cache=True)
def _sensor_work_jit(x0, y0, sensor_range, robot_belief, ground_truth):
    """Cast 720 rays (0.5° increment) from the robot position."""
    sensor_angle_inc = 0.5 / 180.0 * math.pi
    two_pi = 2.0 * math.pi
    sensor_angle = 0.0
    while sensor_angle < two_pi:
        x1 = x0 + math.cos(sensor_angle) * sensor_range
        y1 = y0 + math.sin(sensor_angle) * sensor_range
        _collision_check_jit(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc


def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    """Public API kept for backward compatibility."""
    _collision_check_jit(
        round(float(x0)), round(float(y0)),
        round(float(x1)), round(float(y1)),
        ground_truth, robot_belief,
    )
    return robot_belief


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth):
    """Public API: delegates to the JIT-compiled inner loop."""
    _sensor_work_jit(
        float(robot_position[0]), float(robot_position[1]),
        float(sensor_range),
        robot_belief, ground_truth,
    )
    return robot_belief