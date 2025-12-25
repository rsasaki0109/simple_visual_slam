#pragma once

#include "core/common.h"
#include "core/frame.h"
#include "core/map.h"
#include "tracking/initializer.h"
#include "backend/local_mapping.h"

namespace svslam {

enum class TrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
};

class Tracking {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Tracking>;

    Tracking();

    void setMap(std::shared_ptr<Map> map);
    void setLocalMapping(std::shared_ptr<LocalMapping> local_mapping);
    bool addFrame(Frame::Ptr frame);

    TrackingState state_;
    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;
    
    // Initialization
    Frame::Ptr initial_frame_;
    Initializer::Ptr initializer_;

    // Map reference
    std::shared_ptr<Map> map_;
    std::shared_ptr<LocalMapping> local_mapping_;

    // Motion Model
    // T_cw_current = velocity_ * T_cw_last
    SE3 velocity_; 
    
    int num_tracked_features_ = 0;
    int frames_since_last_kf_ = 0;

private:
    bool track();
    bool initialize();
    bool trackReferenceKeyframe();
    bool trackLocalMap();
    bool needNewKeyframe();
    
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

}
