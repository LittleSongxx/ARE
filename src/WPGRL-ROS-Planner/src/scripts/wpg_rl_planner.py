#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WPGRL-ROS-Planner: Wavelet-Privileged Graph RL planner for autonomous exploration.

Based on ARiADNE-ROS-Planner ROS skeleton, using WPG_RL inference core.
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import rospy
import rospkg
import numpy as np
import torch
import os
import time
from std_msgs.msg import Float32, Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2

from wpg_runtime.agent import Agent
from wpg_runtime.model import PolicyNet
from wpg_runtime.node_manager import NodeManager
from wpg_runtime.utils import MapInfo, get_cell_position_from_coords, check_collision, is_free
from wpg_runtime import parameter
from wpg_runtime.runtime_utils import (
    remap_ros_occupancy_to_wpg,
    load_policy_checkpoint,
    resolve_model_path,
)


def _detect_device(requested="auto"):
    if requested not in ("auto", ""):
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError:
        return "cpu"


class Runner:
    def __init__(self):
        self.map_info = None
        self.device = _detect_device(rospy.get_param('~device', 'auto'))
        self.step = 0

        self.publish_graph = rospy.get_param('~publish_graph', True)

        parameter.CELL_SIZE = rospy.get_param('~map_resolution', parameter.CELL_SIZE)
        parameter.SENSOR_RANGE = rospy.get_param('~sensor_range', parameter.SENSOR_RANGE)
        parameter.UTILITY_RANGE = rospy.get_param('~utility_range_factor', 0.8) * parameter.SENSOR_RANGE
        parameter.MIN_UTILITY = rospy.get_param('~min_utility', parameter.MIN_UTILITY)
        parameter.FRONTIER_CELL_SIZE = rospy.get_param('~frontier_downsample_factor', 2) * parameter.CELL_SIZE
        parameter.NODE_RESOLUTION = rospy.get_param('~node_resolution', parameter.NODE_RESOLUTION)
        parameter.UPDATING_MAP_SIZE = 4 * parameter.SENSOR_RANGE + 4 * parameter.NODE_RESOLUTION

        parameter.THR_TO_WAYPOINT = rospy.get_param('~waypoint_threshold', parameter.THR_TO_WAYPOINT)
        parameter.THR_NEXT_WAYPOINT = rospy.get_param('~next_waypoint_threshold', parameter.THR_NEXT_WAYPOINT)
        parameter.THR_GRAPH_HARD_UPDATE = rospy.get_param('~hard_update_threshold', parameter.THR_GRAPH_HARD_UPDATE)
        parameter.AVOID_OSCILLATION = rospy.get_param('~avoid_waypoint_oscillation', parameter.AVOID_OSCILLATION)
        parameter.ENABLE_SAVE_MODE = rospy.get_param('~enable_save_mode', parameter.ENABLE_SAVE_MODE)
        parameter.ENABLE_CORRIDOR_GRAPH_COMPRESSION = rospy.get_param(
            '~enable_corridor_graph_compression', parameter.ENABLE_CORRIDOR_GRAPH_COMPRESSION)
        parameter.ENABLE_CORRIDOR_EDGE_PRUNING = rospy.get_param(
            '~enable_corridor_edge_pruning', parameter.ENABLE_CORRIDOR_EDGE_PRUNING)

        parameter.ENABLE_GRAPH_RAREFACTION = rospy.get_param(
            '~enable_graph_rarefaction', parameter.ENABLE_GRAPH_RAREFACTION)
        parameter.WAVELET_ADAPTIVE_DTH = rospy.get_param(
            '~wavelet_adaptive_dth', parameter.WAVELET_ADAPTIVE_DTH)
        parameter.WAVELET_DTH_ALPHA = rospy.get_param('~wavelet_dth_alpha', parameter.WAVELET_DTH_ALPHA)
        parameter.WAVELET_DTH_MAX_MULT = rospy.get_param('~wavelet_dth_max_mult', parameter.WAVELET_DTH_MAX_MULT)
        parameter.WAVELET_LOCAL_MAP_SIZE = rospy.get_param('~wavelet_local_map_size', parameter.WAVELET_LOCAL_MAP_SIZE)

        frequency = rospy.get_param('~replanning_frequency', 1.0)
        self.greedy = rospy.get_param('~greedy_action_selection', True)

        self._log_settings()

        self.robot_location = None
        self.robot_cell = None
        self.robot = None
        self.init_agent()
        self.start = None

        self.next_waypoint_list = []
        self.history_waypoint_list = []
        self.next_waypoint = None

        self.done = False
        self.save_mode = False

        rospy.Subscriber('/projected_map', OccupancyGrid, self.get_map_callback, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry, self.get_loc_callback, queue_size=1)

        self.waypoint_pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
        self.run_time_pub = rospy.Publisher('/runtime', Float32, queue_size=1)
        self.edge_pub = rospy.Publisher('/edge', Marker, queue_size=1)
        self.node_pub = rospy.Publisher('/node', PointCloud2, queue_size=1)
        self.frontier_pub = rospy.Publisher('/frontier', PointCloud2, queue_size=1)

        while self.map_info is None or self.robot_location is None:
            pass

        rate = rospy.Rate(20)
        rospy.Timer(rospy.Duration(1 / frequency), self.run)
        try:
            rate.sleep()
            rospy.spin()
        except KeyboardInterrupt:
            pass

    def _log_settings(self):
        rospy.loginfo("=" * 50)
        rospy.loginfo("WPGRL Planner Settings:")
        rospy.loginfo(f"  CELL_SIZE            = {parameter.CELL_SIZE}")
        rospy.loginfo(f"  NODE_RESOLUTION      = {parameter.NODE_RESOLUTION}")
        rospy.loginfo(f"  SENSOR_RANGE         = {parameter.SENSOR_RANGE}")
        rospy.loginfo(f"  NODE_INPUT_DIM       = {parameter.NODE_INPUT_DIM}")
        rospy.loginfo(f"  EMBEDDING_DIM        = {parameter.EMBEDDING_DIM}")
        rospy.loginfo(f"  K_SIZE               = {parameter.K_SIZE}")
        rospy.loginfo(f"  NODE_PADDING_SIZE    = {parameter.NODE_PADDING_SIZE}")
        rospy.loginfo(f"  USE_LF_ATTN_HF_RES  = {parameter.USE_LF_ATTENTION_HF_RESIDUAL}")
        rospy.loginfo(f"  WAVELET_SCALES       = {parameter.WAVELET_SCALES}")
        rospy.loginfo(f"  CORRIDOR_COMPRESSION = {parameter.ENABLE_CORRIDOR_GRAPH_COMPRESSION}")
        rospy.loginfo(f"  CORRIDOR_EDGE_PRUNE  = {parameter.ENABLE_CORRIDOR_EDGE_PRUNING}")
        rospy.loginfo(f"  GRAPH_RAREFACTION    = {parameter.ENABLE_GRAPH_RAREFACTION}")
        rospy.loginfo(f"  WAVELET_ADAPTIVE_DTH = {parameter.WAVELET_ADAPTIVE_DTH}")
        rospy.loginfo(f"  DEVICE               = {self.device}")
        rospy.loginfo("=" * 50)

    def get_map_callback(self, msg):
        delta = msg.info.resolution
        map_origin_x = msg.info.origin.position.x
        map_origin_y = msg.info.origin.position.y

        map_width = msg.info.width
        map_height = msg.info.height
        ros_map = np.array(msg.data).reshape(map_height, map_width).astype(np.int8)

        wpg_map = remap_ros_occupancy_to_wpg(ros_map)

        pad_size = int(parameter.NODE_RESOLUTION // parameter.CELL_SIZE + 1)
        processed_map = np.pad(
            wpg_map,
            ((pad_size, pad_size), (pad_size, pad_size)),
            'constant',
            constant_values=parameter.UNKNOWN,
        )
        map_origin_x -= delta * pad_size
        map_origin_y -= delta * pad_size

        self.map_info = MapInfo(processed_map, map_origin_x, map_origin_y, delta)

    def get_loc_callback(self, msg):
        if self.map_info is None:
            return
        self.robot_location = np.around(
            np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]), 1
        )
        if self.start is None:
            x = np.array([
                (self.robot_location[0] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION,
                (self.robot_location[0] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION,
            ])
            y = np.array([
                (self.robot_location[1] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION,
                (self.robot_location[1] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION,
            ])
            t1, t2 = np.meshgrid(x, y)
            candidate_starts = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
            dis_robot = np.linalg.norm(candidate_starts - self.robot_location, axis=1)
            sorted_candidate_starts = candidate_starts[np.argsort(dis_robot)]

            for start in sorted_candidate_starts:
                if is_free(start, self.map_info):
                    self.start = start
                    break

            if self.start is None:
                rospy.logwarn("Cannot find valid start point, retrying next callback")
                return

            self.start = np.around(self.start, 1)
            self.robot.node_manager = NodeManager(plot=self.publish_graph)
            self.robot.node_manager.add_node_to_dict(self.start, set(), None)
            rospy.loginfo(f"Initialized quad tree at {self.start}")
            rospy.loginfo(f"Initialized robot location at {self.robot_location}")

        self.robot_cell = get_cell_position_from_coords(self.robot_location, self.map_info)

    def waypoint_wrapper(self, loc):
        way_point = PointStamped()
        way_point.header.frame_id = "map"
        way_point.header.stamp = rospy.Time.now()
        way_point.point.x = loc[0]
        way_point.point.y = loc[1]
        return way_point

    def init_agent(self):
        policy_net = PolicyNet(
            parameter.NODE_INPUT_DIM,
            parameter.EMBEDDING_DIM,
            use_lf_attention_hf_residual=parameter.USE_LF_ATTENTION_HF_RESIDUAL,
            wavelet_scales=parameter.WAVELET_SCALES,
            wavelet_fuse_dim=parameter.WAVELET_FUSE_DIM,
            wavelet_lf_qk=parameter.WAVELET_LF_QK,
        ).to(self.device)

        pkg_path = rospkg.RosPack().get_path('wpg_rl_planner')
        rosparam_model = rospy.get_param('~model_path', '')
        ckpt_path = resolve_model_path(pkg_path, rosparam_model)
        rospy.loginfo(f"Loading checkpoint from: {ckpt_path}")
        load_policy_checkpoint(policy_net, ckpt_path, self.device)
        rospy.loginfo("Checkpoint loaded successfully")

        self.robot = Agent(policy_net, self.device, self.publish_graph)

    def run(self, event=None):
        t1 = time.time()
        if self.done:
            return

        if self.save_mode:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                return
            else:
                if len(self.next_waypoint_list) > 0:
                    next_waypoint = self.next_waypoint_list.pop(0)
                    while (
                        check_collision(self.robot_location, np.array(next_waypoint), self.map_info) is False
                        and np.linalg.norm(self.robot_location - np.array(next_waypoint)) < (
                            parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION
                        )
                        and len(self.next_waypoint_list) > 0
                    ):
                        next_waypoint = self.next_waypoint_list.pop(0)
                    self.next_waypoint = next_waypoint
                    self.history_waypoint_list.append((self.next_waypoint[0], self.next_waypoint[1]))
                    self.waypoint_pub.publish(self.waypoint_wrapper(self.next_waypoint))
                    run_time = Float32()
                    run_time.data = time.time() - t1
                    self.run_time_pub.publish(run_time)
                    return
                else:
                    self.save_mode = False
                    rospy.logwarn("Switch back to RL")

        if parameter.AVOID_OSCILLATION and len(self.history_waypoint_list) > 4:
            if (
                self.history_waypoint_list[-1] == self.history_waypoint_list[-3]
                and self.history_waypoint_list[-2] == self.history_waypoint_list[-4]
            ):
                self.next_waypoint_list = []
                if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                    return

        if len(self.next_waypoint_list) > 0:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                pass
            else:
                self.robot_location = self.next_waypoint
                self.next_waypoint = self.next_waypoint_list.pop(0)
                self.waypoint_pub.publish(self.waypoint_wrapper(self.next_waypoint))
        self.next_waypoint_list = []

        self.robot.node_manager.check_valid_node(self.robot_location, self.map_info)

        robot_node_location = self.robot_location
        if self.robot_location[0] != self.start[0] or self.robot_location[1] != self.start[1]:
            if self.robot.node_manager.nodes_dict.__len__() == 0:
                robot_node_location = self.start
            else:
                nearest_node = self.robot.node_manager.nodes_dict.nearest_neighbors(
                    self.robot_location.tolist(), 1
                )[0]
                node_coords = nearest_node.data.coords
                robot_node_location = node_coords

        self.robot.update_planning_state(self.map_info, robot_node_location)

        if self.robot.utility is None or sum(self.robot.utility) == 0:
            g = "\033[92m"
            n = "\033[0m"
            rospy.loginfo(f"{g}Exploration Completed{n}")
            self.done = True
            run_time = Float32()
            run_time.data = 0
            self.run_time_pub.publish(run_time)
            return

        observation = self.robot.get_observation()

        next_location, next_node_index = self.robot.select_next_waypoint(observation, greedy=self.greedy)

        self.next_waypoint_list.append(next_location)
        if len(self.history_waypoint_list) > 0:
            if (next_location[0], next_location[1]) != self.history_waypoint_list[-1]:
                self.history_waypoint_list.append((next_location[0], next_location[1]))
        else:
            self.history_waypoint_list.append((next_location[0], next_location[1]))

        next_node = self.robot.node_manager.nodes_dict.find(next_location.tolist())
        if next_node is not None and next_node.data.utility == 0:
            try:
                next_observation = self.robot.get_next_observation(next_node_index, observation)
                next_next_location, _ = self.robot.select_next_waypoint(next_observation, greedy=self.greedy)
                if np.linalg.norm(next_location - self.robot_location) < parameter.NODE_RESOLUTION:
                    self.next_waypoint_list = []
                self.next_waypoint_list.append(next_next_location)
            except Exception:
                pass

        t4 = time.time()

        if parameter.ENABLE_SAVE_MODE:
            if self.detect_waypoint_loop():
                self.next_waypoint_list = self.robot.node_manager.path_to_nearest_frontier
                if len(self.next_waypoint_list) > 0:
                    self.save_mode = True
                    rospy.logwarn("Switch to save mode")

        if not self.next_waypoint_list:
            return
        self.next_waypoint = self.next_waypoint_list.pop(0)
        self.waypoint_pub.publish(self.waypoint_wrapper(self.next_waypoint))

        run_time = Float32()
        run_time.data = t4 - t1
        self.run_time_pub.publish(run_time)

        self.step += 1
        if self.publish_graph:
            self.visualize_graph()

    def detect_waypoint_loop(self, max_length=6):
        if len(self.history_waypoint_list) < max_length:
            return False

        waypoint_list_to_check = self.history_waypoint_list[-max_length:]
        loop = []
        for i, waypoint in enumerate(waypoint_list_to_check[:-1]):
            if waypoint == waypoint_list_to_check[-1]:
                loop = waypoint_list_to_check[i:]

        if loop:
            loop_length = len(loop)
            if len(self.history_waypoint_list) < 2 * loop_length + 1:
                return False
            waypoint_list_to_check2 = self.history_waypoint_list[
                -max_length - loop_length + 1 : -loop_length + 1
            ]
            loop2 = []
            for i, waypoint in enumerate(waypoint_list_to_check2[:-1]):
                if waypoint == waypoint_list_to_check2[-1]:
                    loop2 = waypoint_list_to_check2[i:]
                    break
            if loop2:
                return True
        return False

    def visualize_graph(self):
        if self.robot.node_coords is None or self.robot.adjacent_matrix is None:
            return

        edges = Marker()
        edges.header.frame_id = 'map'
        edges.header.stamp = rospy.Time.now()
        edges.type = Marker.LINE_LIST
        edges.scale.x = 0.1
        edges.color.r = 0.0
        edges.color.g = 0.6
        edges.color.b = 0.0
        edges.color.a = 1.0
        edges.pose.orientation.w = 1.0

        for i, coords in enumerate(self.robot.node_coords):
            neighbors = np.argwhere(self.robot.adjacent_matrix[i] == 0).reshape(-1)
            for j in neighbors:
                if j <= i:
                    continue
                neighbor_coords = self.robot.node_coords[j]
                start = Point()
                start.x = coords[0]
                start.y = coords[1]
                end = Point()
                end.x = neighbor_coords[0]
                end.y = neighbor_coords[1]
                edges.points.append(start)
                edges.points.append(end)
        self.edge_pub.publish(edges)

        nodes = []
        for node_coords, utility in zip(self.robot.node_coords, self.robot.utility):
            nodes.append((node_coords[0], node_coords[1], 0.0, float(utility)))
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        nodes_msg = point_cloud2.create_cloud(header, fields, nodes)
        self.node_pub.publish(nodes_msg)

        frontiers = []
        for frontier in self.robot.frontier:
            frontiers.append((frontier[0], frontier[1], 0))
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        frontiers_msg = point_cloud2.create_cloud(header, fields, frontiers)
        self.frontier_pub.publish(frontiers_msg)


if __name__ == '__main__':
    rospy.init_node('wpg_rl_planner', anonymous=True)
    rl_runner = Runner()
