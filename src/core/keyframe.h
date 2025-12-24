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
