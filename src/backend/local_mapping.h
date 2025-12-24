#pragma once

#include "core/common.h"
#include "core/map.h"
#include "core/keyframe.h"
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>

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
    
private:
    void processNewKeyframe();
    void createNewMapPoints();
    void mapPointCulling();
    void optimization();
    
    // Check if there are keyframes in the queue
    bool checkNewKeyframes();
    
    Map::Ptr map_;
    
    std::deque<Keyframe::Ptr> new_keyframes_;
    Keyframe::Ptr current_processed_kf_;
    
    std::shared_ptr<LoopClosing> loop_closing_;
    
    std::mutex mutex_new_keyframes_;
    std::condition_variable cv_new_keyframes_;
    
    bool stop_requested_ = false;
};

}
