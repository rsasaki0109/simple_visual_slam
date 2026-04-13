# ROS2 Wrapper

This directory contains a basic ROS 2 Jazzy package named `simple_visual_slam_ros2`. The package builds a `slam_node` executable with `ament_cmake` and links it against the existing `svslam_core` target by adding the repository root as a CMake subdirectory.

## Build

1. Source ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
```

2. Put this repository inside a colcon workspace `src/` directory if it is not already there.

3. Build only the ROS 2 wrapper package:

```bash
colcon build --packages-select simple_visual_slam_ros2
```

The ROS 2 package disables inherited tests before pulling in the root SimpleVisualSLAM CMake project, so the wrapper only builds the `slam_node` target and the `svslam_core` library it depends on.

## Run

Launch the node with the default camera topic:

```bash
source install/setup.bash
ros2 launch simple_visual_slam_ros2 slam.launch.py
```

Enable depth input and override topics:

```bash
ros2 launch simple_visual_slam_ros2 slam.launch.py \
  camera_topic:=/camera/image_raw \
  depth_topic:=/camera/depth \
  use_depth:=true \
  max_features:=2500
```

The node subscribes to:

- `camera_topic` for the main image stream
- `camera_info` derived from the camera topic namespace
- `depth_topic` when `use_depth:=true`

The node publishes:

- `odom` as `nav_msgs/Odometry`
- `path` as `nav_msgs/Path`
- `landmarks` as `sensor_msgs/PointCloud2`
- a TF transform from `map` to the incoming camera frame

## Notes

- This is a minimal ROS 2 integration. It drives `svslam::Tracking` directly and runs local mapping synchronously inside the node callback to keep the wrapper simple.
- The `vocab_path` parameter is declared in the launch file for future loop-closing integration, but this basic node does not currently consume it.
- No top-level CMake files in the repository need to be modified for this wrapper.
