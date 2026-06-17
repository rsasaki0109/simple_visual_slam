#pragma once

#include "core/common.h"
#include "core/camera.h"
#include <opencv2/features2d.hpp>

namespace svslam {

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Frame>;

    Frame() {}
    Frame(unsigned long id, double timestamp, Camera::Ptr camera, const cv::Mat& image);

    void setPose(const SE3& pose);
    SE3 getPose() const;

    // Feature extraction
    void extractORB(const cv::Ptr<cv::Feature2D>& detector);

    unsigned long id_;
    double timestamp_;
    Camera::Ptr camera_;
    cv::Mat image_;
    
    // Pose: T_world_camera (Camera to World) or T_cw (World to Camera)
    // Let's use T_cw (World -> Camera) as is common in ORB-SLAM
    SE3 T_cw_;

    // VIO state (scaffolding for Stage 0b preintegration; not yet used in BA).
    // velocity_ is in the world frame (m/s). accel_bias_ / gyro_bias_ are
    // IMU biases in the sensor frame (m/s^2, rad/s). has_velocity_ is set
    // once the tracking pipeline produces a usable velocity estimate.
    Vec3 velocity_ = Vec3::Zero();
    Vec3 accel_bias_ = Vec3::Zero();
    Vec3 gyro_bias_ = Vec3::Zero();
    bool has_velocity_ = false;

    // Depth image (CV_16UC1 in mm for sensor depth, or CV_32FC1 in meters for DL depth)
    cv::Mat depth_image_;
    bool depth_is_metric_ = true;  // true for sensor/metric DL depth, false for relative DL depth
    bool depth_is_learned_ = false;  // true if depth came from an ONNX model (used to soften BA trust)

    // Get depth at pixel (u,v) in meters. Returns <= 0 if invalid.
    float getDepth(float u, float v) const;

    // Back-project pixel with known depth to 3D world point
    Vec3 backprojectWithDepth(const cv::KeyPoint& kp, float depth_m) const;

    // Snapshot landmarks_ under mutex_ for safe cross-thread iteration.
    // Writers on tracking thread must take mutex_ around landmarks_ writes;
    // readers on other threads (LocalMapping onBACompleted path) should call
    // this instead of iterating landmarks_ directly.
    std::vector<std::shared_ptr<Landmark>> snapshotLandmarks() const;

    // Features
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    // Map points associated with features.
    // Writes must be guarded by mutex_ once this Frame has been published as
    // Tracking::current_frame_ (LocalMapping::onBACompleted reads it via the
    // tracking BA callback). See snapshotLandmarks() for reads.
    std::vector<std::shared_ptr<Landmark>> landmarks_;
    
    // Grid for fast search (optional, but good for requirements)
    // Skipping grid implementation for now to keep it minimal, 
    // but reserving member if needed or just using brute force for now.
    
    mutable std::mutex mutex_;
};

}
