#pragma once

#include "core/common.h"
#include "core/frame.h"
#include "core/map.h"
#include "tracking/initializer.h"
#include "backend/local_mapping.h"
#include <mutex>

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

    // Callback from LocalMapping when BA is completed
    void onBACompleted();

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
    int consecutive_tracking_failures_ = 0;
    Keyframe::Ptr reference_keyframe_;

private:
    bool track();
    bool initialize();
    bool trackReferenceKeyframe();
    bool trackLocalMap();
    bool needNewKeyframe();
    void recomputeCurrentPose();
    bool relocalize();  // Attempt to recover from tracking loss

    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::mutex pose_mutex_;  // For thread-safe pose updates

    int lost_frame_count_ = 0;
    static constexpr int max_lost_frames_ = 30;  // Max frames before giving up
};

}
