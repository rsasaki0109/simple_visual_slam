#pragma once

#include "core/common.h"
#include "core/map.h"
#include "core/keyframe.h"
#include <mutex>
#include <thread>
#include <deque>
#include <condition_variable>

#include <DBoW2/DBoW2.h>

namespace svslam {

// Define Vocabulary type
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbDatabase;

class LoopClosing {
public:
    using Ptr = std::shared_ptr<LoopClosing>;

    LoopClosing(Map::Ptr map, const std::string& vocab_path);
    
    // Main loop
    void run();
    
    // Input
    void insertKeyframe(Keyframe::Ptr kf);
    
    void requestStop();
    
private:
    bool checkNewKeyframes();
    void processNewKeyframe();
    bool detectLoop(const DBoW2::BowVector& bow_vec);
    bool computeSim3();
    void correctLoop();
    
    Map::Ptr map_;
    std::string vocab_path_;
    std::shared_ptr<OrbVocabulary> vocab_;
    std::shared_ptr<OrbDatabase> db_;
    
    // Queue
    std::deque<Keyframe::Ptr> new_keyframes_;
    std::mutex mutex_new_keyframes_;
    std::condition_variable cv_new_keyframes_;
    
    Keyframe::Ptr current_processed_kf_;

    std::vector<Keyframe::Ptr> db_keyframes_;
    int min_loop_interval_kf_ = 30;
    int max_loop_candidates_ = 4;
    double min_loop_score_ = 0.05;
    
    bool stop_requested_ = false;
};

}
