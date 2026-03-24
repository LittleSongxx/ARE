cmake_minimum_required(VERSION 3.0.2)
project(wpg_rl_planner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
  octomap_ros
  octomap_msgs
  tf2_ros
  tf2_geometry_msgs
)

find_package(PCL REQUIRED)
find_package(octomap REQUIRED)
find_package(pybind11 QUIET)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs nav_msgs
                 pcl_ros pcl_conversions octomap_ros octomap_msgs
  DEPENDS PCL octomap
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

# ---- map_handler node ----
add_executable(wpg_map_handler src/map_handler.cpp)
set_target_properties(wpg_map_handler PROPERTIES OUTPUT_NAME map_handler)
target_link_libraries(wpg_map_handler
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

# ---- terrain_segmentation node ----
add_executable(wpg_terrain_seg src/terrain_segmentation.cpp)
set_target_properties(wpg_terrain_seg PROPERTIES OUTPUT_NAME terrain_segmentation)
target_link_libraries(wpg_terrain_seg
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# ---- collision_checker pybind11 module (optional) ----
if(pybind11_FOUND)
  pybind11_add_module(wpg_collision_checker src/collision_checker.cpp)
  target_link_libraries(wpg_collision_checker PRIVATE
    ${catkin_LIBRARIES}
    pybind11::module
  )
  set_target_properties(wpg_collision_checker PROPERTIES
    OUTPUT_NAME collision_checker
    LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
  )
  install(TARGETS wpg_collision_checker
    LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  )
else()
  message(WARNING "pybind11 not found – collision_checker module will NOT be built. "
                  "Install pybind11-dev to enable C++ accelerated collision checking.")
endif()
