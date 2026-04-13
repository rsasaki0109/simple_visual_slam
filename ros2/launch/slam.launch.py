from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_topic = LaunchConfiguration("camera_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    use_depth = LaunchConfiguration("use_depth")
    vocab_path = LaunchConfiguration("vocab_path")
    max_features = LaunchConfiguration("max_features")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_topic",
                default_value="/camera/image_raw",
                description="Input camera image topic.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/depth",
                description="Optional depth image topic.",
            ),
            DeclareLaunchArgument(
                "use_depth",
                default_value="false",
                description="Subscribe to the depth topic and attach depth to frames.",
            ),
            DeclareLaunchArgument(
                "vocab_path",
                default_value="",
                description="Reserved vocabulary path parameter for future loop-closing integration.",
            ),
            DeclareLaunchArgument(
                "max_features",
                default_value="2000",
                description="Maximum ORB features extracted per frame.",
            ),
            Node(
                package="simple_visual_slam_ros2",
                executable="slam_node",
                name="slam_node",
                output="screen",
                parameters=[
                    {
                        "camera_topic": camera_topic,
                        "depth_topic": depth_topic,
                        "use_depth": use_depth,
                        "vocab_path": vocab_path,
                        "max_features": max_features,
                    }
                ],
            ),
        ]
    )
