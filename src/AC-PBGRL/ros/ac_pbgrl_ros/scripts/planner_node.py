#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Trigger, TriggerResponse

from ac_pbgrl_ros.graph_builder import OccupancyGraphBuilder, stable_id
from ac_pbgrl_ros.inference import ONNXExplorer


class PlannerNode:
    def __init__(self):
        model_path = Path(rospy.get_param("~model_path"))
        metadata_path = Path(rospy.get_param("~metadata_path", str(model_path.with_suffix(".json"))))
        metadata = json.loads(metadata_path.read_text())
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.lock = threading.RLock()
        self.map_message = None
        self.robot_xy = None
        self.builder = OccupancyGraphBuilder(
            nodes=int(metadata["nodes"]),
            candidates=int(metadata["candidates"]),
            node_resolution_m=float(rospy.get_param("~node_resolution", 4.0)),
            sensor_range_m=float(rospy.get_param("~sensor_range", 16.0)),
            local_budget=int(rospy.get_param("~local_budget", min(192, int(metadata["nodes"])))),
            region_size_m=float(rospy.get_param("~region_size", 16.0)),
        )
        self.explorer = ONNXExplorer(model_path, metadata=metadata)
        self.publisher = rospy.Publisher(rospy.get_param("~waypoint_topic", "/way_point"), PointStamped, queue_size=1)
        rospy.Subscriber(rospy.get_param("~map_topic", "/projected_map"), OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~odometry_topic", "/state_estimation"), Odometry, self.odometry_callback, queue_size=1)
        rospy.Service("~reset", Trigger, self.reset_callback)
        rate = float(rospy.get_param("~planning_rate", 1.0))
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(rate, 1e-3)), self.plan)

    def map_callback(self, message):
        with self.lock:
            self.map_message = message

    def odometry_callback(self, message):
        with self.lock:
            self.robot_xy = (message.pose.pose.position.x, message.pose.pose.position.y)

    def reset_callback(self, request):
        del request
        self.builder.reset()
        self.explorer.reset()
        return TriggerResponse(success=True, message="AC-PBGRL temporal state cleared")

    def plan(self, event):
        del event
        with self.lock:
            message, robot_xy = self.map_message, self.robot_xy
        if message is None or robot_xy is None:
            return
        grid = np.asarray(message.data, dtype=np.int16).reshape(message.info.height, message.info.width)
        origin = (message.info.origin.position.x, message.info.origin.position.y)
        started = time.perf_counter()
        try:
            self.builder.mark_visited(*robot_xy)
            self.explorer.retire(stable_id(*robot_xy))
            graph = self.builder.build(grid, message.info.resolution, origin, robot_xy)
            if not graph.feeds["candidate_mask"].any():
                rospy.logwarn_throttle(5.0, "AC-PBGRL found no executable candidate")
                return
            slot, _ = self.explorer.select(graph)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "AC-PBGRL planning failed: %s", exc)
            return
        waypoint = graph.candidate_xy[slot]
        output = PointStamped()
        output.header.stamp = rospy.Time.now()
        output.header.frame_id = self.frame_id
        output.point.x = float(waypoint[0])
        output.point.y = float(waypoint[1])
        output.point.z = 0.0
        self.publisher.publish(output)
        rospy.logdebug("AC-PBGRL planning latency %.2f ms", (time.perf_counter() - started) * 1000.0)


if __name__ == "__main__":
    rospy.init_node("ac_pbgrl_planner")
    PlannerNode()
    rospy.spin()
