#pragma once

#include "core/common.h"
#include "core/map.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include <deque>
#include <list>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

namespace svslam {

class LoopClosing; // Forward decl

class LocalMapping {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<LocalMapping>;

    LocalMapping(Map::Ptr map);

    void setLoopClosing(std::shared_ptr<LoopClosing> loop_closing);

    void insertKeyframe(Keyframe::Ptr kf);
    void run(); // Main loop
    void requestStop();

    // Callback for BA completion notification
    std::function<void()> on_ba_completed_;

private:
    using RecentLandmark = std::pair<Landmark::Ptr, unsigned long>;

    void processNewKeyframe();
    void createNewMapPoints();
    void mapPointCulling();
    void optimization();
    void removeLandmark(const Landmark::Ptr& lm);

    // Check if there are keyframes in the queue
    bool checkNewKeyframes();

    Map::Ptr map_;

    std::deque<Keyframe::Ptr> new_keyframes_;
    Keyframe::Ptr current_processed_kf_;
    std::list<RecentLandmark> recent_landmarks_;
    unsigned long processed_keyframe_count_ = 0;

    std::shared_ptr<LoopClosing> loop_closing_;

    std::mutex mutex_new_keyframes_;
    std::condition_variable cv_new_keyframes_;

    bool stop_requested_ = false;
};

}
