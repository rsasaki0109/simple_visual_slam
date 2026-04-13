cmake_minimum_required(VERSION 3.16)
project(simple_visual_slam_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(BUILD_TESTING OFF CACHE BOOL "Disable tests while building the ROS2 wrapper" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/simple_visual_slam_core EXCLUDE_FROM_ALL)

add_executable(slam_node
  src/slam_node.cc
)

target_compile_features(slam_node PRIVATE cxx_std_17)
target_link_libraries(slam_node svslam_core)

ament_target_dependencies(slam_node
  cv_bridge
  geometry_msgs
  image_transport
  nav_msgs
  rclcpp
  sensor_msgs
  tf2_ros
)

install(TARGETS slam_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
