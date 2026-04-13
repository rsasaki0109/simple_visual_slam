#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/features2d.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include "backend/local_mapping.h"
#include "core/camera.h"
#include "core/frame.h"
#include "core/landmark.h"
#include "core/map.h"
#include "tracking/tracking.h"

namespace {

constexpr char kMapFrame[] = "map";
constexpr char kDefaultCameraFrame[] = "camera";
constexpr double kDepthSyncToleranceSec = 0.10;

std::string derive_camera_info_topic(const std::string& camera_topic) {
    const std::size_t last_slash = camera_topic.find_last_of('/');
    if (last_slash == std::string::npos) {
        return "camera_info";
    }
    return camera_topic.substr(0, last_slash + 1) + "camera_info";
}

geometry_msgs::msg::Quaternion to_geometry_quaternion(const Eigen::Quaterniond& q) {
    geometry_msgs::msg::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
}

}  // namespace

class SlamNode : public rclcpp::Node {
public:
    SlamNode()
        : rclcpp::Node("slam_node"),
          tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)) {
        camera_topic_ = declare_parameter<std::string>("camera_topic", "/camera/image_raw");
        depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/depth");
        use_depth_ = declare_parameter<bool>("use_depth", false);
        vocab_path_ = declare_parameter<std::string>("vocab_path", "");
        max_features_ = declare_parameter<int>("max_features", 2000);
        camera_info_topic_ = derive_camera_info_topic(camera_topic_);

        if (max_features_ <= 0) {
            RCLCPP_WARN(get_logger(), "max_features must be positive, falling back to 2000");
            max_features_ = 2000;
        }
        if (!vocab_path_.empty()) {
            RCLCPP_INFO(
                get_logger(),
                "Parameter 'vocab_path' is set to '%s' but loop closing is not enabled in this basic ROS2 node",
                vocab_path_.c_str());
        }

        orb_ = cv::ORB::create(max_features_);

        map_ = std::make_shared<svslam::Map>();
        local_mapping_ = std::make_shared<svslam::LocalMapping>(map_);
        tracker_ = std::make_shared<svslam::Tracking>();
        tracker_->setMap(map_);
        tracker_->setLocalMapping(local_mapping_);

        const std::weak_ptr<svslam::Tracking> tracker_weak = tracker_;
        local_mapping_->on_ba_completed_ = [tracker_weak]() {
            if (const auto tracker = tracker_weak.lock()) {
                tracker->onBACompleted();
            }
        };

        path_msg_.header.frame_id = kMapFrame;

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odom", 10);
        path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
        landmarks_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("landmarks", 10);

        const auto sensor_qos = rclcpp::SensorDataQoS();
        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_,
            sensor_qos,
            std::bind(&SlamNode::camera_info_callback, this, std::placeholders::_1));

        image_sub_ = image_transport::create_subscription(
            this,
            camera_topic_,
            std::bind(&SlamNode::image_callback, this, std::placeholders::_1),
            "raw",
            rmw_qos_profile_sensor_data);

        if (use_depth_) {
            depth_sub_ = image_transport::create_subscription(
                this,
                depth_topic_,
                std::bind(&SlamNode::depth_callback, this, std::placeholders::_1),
                "raw",
                rmw_qos_profile_sensor_data);
        }

        RCLCPP_INFO(
            get_logger(),
            "Listening on image topic '%s', camera info topic '%s'%s",
            camera_topic_.c_str(),
            camera_info_topic_.c_str(),
            use_depth_ ? ", and depth topic enabled" : "");
    }

private:
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
        const double fx = msg->k[0] > 0.0 ? msg->k[0] : msg->p[0];
        const double fy = msg->k[4] > 0.0 ? msg->k[4] : msg->p[5];
        const double cx = msg->k[2] != 0.0 ? msg->k[2] : msg->p[2];
        const double cy = msg->k[5] != 0.0 ? msg->k[5] : msg->p[6];

        if (fx <= 0.0 || fy <= 0.0) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000, "Ignoring CameraInfo without valid focal lengths");
            return;
        }

        double k1 = 0.0;
        double k2 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;
        double k3 = 0.0;

        if (msg->d.size() > 0) k1 = msg->d[0];
        if (msg->d.size() > 1) k2 = msg->d[1];
        if (msg->d.size() > 2) p1 = msg->d[2];
        if (msg->d.size() > 3) p2 = msg->d[3];
        if (msg->d.size() > 4) k3 = msg->d[4];

        std::lock_guard<std::mutex> lock(data_mutex_);
        camera_ = std::make_shared<svslam::Camera>(fx, fy, cx, cy, k1, k2, p1, p2, k3);
    }

    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_depth_msg_ = msg;
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> process_lock(process_mutex_);

        svslam::Camera::Ptr camera;
        sensor_msgs::msg::Image::ConstSharedPtr depth_msg;
        {
            std::lock_guard<std::mutex> data_lock(data_mutex_);
            camera = camera_;
            if (use_depth_ && latest_depth_msg_ != nullptr && is_depth_synchronized(msg, latest_depth_msg_)) {
                depth_msg = latest_depth_msg_;
            }
        }

        if (!camera) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "Waiting for CameraInfo on '%s' before processing images",
                camera_info_topic_.c_str());
            return;
        }

        cv_bridge::CvImagePtr image_ptr;
        try {
            image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        } catch (const cv_bridge::Exception& ex) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "Failed to convert input image to MONO8: %s",
                ex.what());
            return;
        }

        auto frame = std::make_shared<svslam::Frame>(
            next_frame_id_++,
            rclcpp::Time(msg->header.stamp).seconds(),
            camera,
            image_ptr->image);

        if (depth_msg != nullptr) {
            try {
                const auto depth_ptr = cv_bridge::toCvCopy(depth_msg);
                const int depth_type = depth_ptr->image.type();
                if (depth_type == CV_16UC1 || depth_type == CV_32FC1) {
                    frame->depth_image_ = depth_ptr->image.clone();
                    frame->depth_is_metric_ = true;
                } else {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(),
                        *get_clock(),
                        5000,
                        "Unsupported depth encoding '%s'; expected 16UC1 or 32FC1",
                        depth_msg->encoding.c_str());
                }
            } catch (const cv_bridge::Exception& ex) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    5000,
                    "Failed to convert depth image: %s",
                    ex.what());
            }
        } else if (use_depth_) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "Depth is enabled but no synchronized depth frame is available on '%s'",
                depth_topic_.c_str());
        }

        frame->extractORB(orb_);
        tracker_->addFrame(frame);
        local_mapping_->processPendingWork();

        publish_outputs(msg->header.stamp, msg->header.frame_id);
    }

    bool is_depth_synchronized(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) const {
        const rclcpp::Time image_time(image_msg->header.stamp);
        const rclcpp::Time depth_time(depth_msg->header.stamp);
        return std::abs((image_time - depth_time).seconds()) <= kDepthSyncToleranceSec;
    }

    void publish_outputs(
        const builtin_interfaces::msg::Time& stamp,
        const std::string& image_frame_id) {
        publish_landmarks(stamp);

        if (!tracker_->current_frame_ || tracker_->state_ != svslam::TrackingState::OK) {
            return;
        }

        const svslam::SE3 T_wc = tracker_->current_frame_->getPose().inverse();
        const Eigen::Vector3d position = T_wc.translation();
        Eigen::Quaterniond orientation(T_wc.unit_quaternion());
        orientation.normalize();

        const std::string child_frame_id =
            image_frame_id.empty() ? kDefaultCameraFrame : image_frame_id;

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = kMapFrame;
        odom_msg.child_frame_id = child_frame_id;
        odom_msg.pose.pose.position.x = position.x();
        odom_msg.pose.pose.position.y = position.y();
        odom_msg.pose.pose.position.z = position.z();
        odom_msg.pose.pose.orientation = to_geometry_quaternion(orientation);
        odom_pub_->publish(odom_msg);

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = odom_msg.header;
        pose_msg.pose = odom_msg.pose.pose;
        path_msg_.header.stamp = stamp;
        path_msg_.poses.push_back(pose_msg);
        path_pub_->publish(path_msg_);

        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = kMapFrame;
        tf_msg.child_frame_id = child_frame_id;
        tf_msg.transform.translation.x = position.x();
        tf_msg.transform.translation.y = position.y();
        tf_msg.transform.translation.z = position.z();
        tf_msg.transform.rotation = to_geometry_quaternion(orientation);
        tf_broadcaster_->sendTransform(tf_msg);
    }

    void publish_landmarks(const builtin_interfaces::msg::Time& stamp) {
        std::vector<svslam::Landmark::Ptr> landmarks;
        {
            std::lock_guard<std::mutex> lock(map_->mutex_);
            const auto& all_landmarks = map_->getAllLandmarks();
            landmarks.reserve(all_landmarks.size());
            for (const auto& entry : all_landmarks) {
                const auto& landmark = entry.second;
                if (!landmark || landmark->isBad()) {
                    continue;
                }
                landmarks.push_back(landmark);
            }
        }

        std::vector<svslam::Vec3> points;
        points.reserve(landmarks.size());
        for (const auto& landmark : landmarks) {
            points.push_back(landmark->getPos());
        }

        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = stamp;
        cloud_msg.header.frame_id = kMapFrame;

        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(points.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        for (const auto& point : points) {
            *iter_x = static_cast<float>(point.x());
            *iter_y = static_cast<float>(point.y());
            *iter_z = static_cast<float>(point.z());
            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        landmarks_pub_->publish(cloud_msg);
    }

    std::string camera_topic_;
    std::string camera_info_topic_;
    std::string depth_topic_;
    std::string vocab_path_;
    bool use_depth_ = false;
    int max_features_ = 2000;
    unsigned long next_frame_id_ = 0;

    std::mutex data_mutex_;
    std::mutex process_mutex_;

    svslam::Camera::Ptr camera_;
    sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg_;

    cv::Ptr<cv::Feature2D> orb_;
    svslam::Map::Ptr map_;
    svslam::LocalMapping::Ptr local_mapping_;
    svslam::Tracking::Ptr tracker_;

    nav_msgs::msg::Path path_msg_;

    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr landmarks_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamNode>());
    rclcpp::shutdown();
    return 0;
}
