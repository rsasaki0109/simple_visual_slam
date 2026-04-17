#pragma once

#include "core/common.h"
#include "core/frame.h"

namespace svslam {

class Keyframe : public std::enable_shared_from_this<Keyframe> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Keyframe>;

    Keyframe(Frame::Ptr frame);

    unsigned long id_;
    double timestamp_;
    Camera::Ptr camera_;
    
    SE3 T_cw_;

    // Depth image (copied from Frame)
    cv::Mat depth_image_;
    bool depth_is_metric_ = true;
    bool depth_is_learned_ = false;  // true if depth came from an ONNX model

    // Gravity direction in camera frame (from accelerometer, if available)
    // Used for gravity constraint in BA (constrains roll/pitch)
    Vec3 gravity_in_camera_ = Vec3::Zero();
    bool has_gravity_ = false;

    float getDepth(float u, float v) const;

    // Features (copied from Frame to be immutable/independent)
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    // Map points
    std::vector<std::shared_ptr<Landmark>> landmarks_;

    // Covisibility Graph
    // Keyframe -> weight (number of shared points)
    std::map<std::shared_ptr<Keyframe>, int> connected_keyframes_;
    
    void updateConnections();
    void addConnection(std::shared_ptr<Keyframe> kf, int weight);
    std::vector<std::shared_ptr<Keyframe>> getBestCovisibilityKeyframes(int N);
    
    std::mutex mutex_;
};

}
