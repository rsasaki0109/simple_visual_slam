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

    // Features
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    // Map points associated with features
    std::vector<std::shared_ptr<Landmark>> landmarks_;
    
    // Grid for fast search (optional, but good for requirements)
    // Skipping grid implementation for now to keep it minimal, 
    // but reserving member if needed or just using brute force for now.
    
    std::mutex mutex_;
};

}
