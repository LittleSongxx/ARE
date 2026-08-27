cmake_minimum_required(VERSION 3.0.2)
project(ac_pbgrl_ros)

find_package(catkin REQUIRED COMPONENTS rospy nav_msgs geometry_msgs std_srvs)
catkin_python_setup()
catkin_package()

catkin_install_python(
  PROGRAMS scripts/planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
