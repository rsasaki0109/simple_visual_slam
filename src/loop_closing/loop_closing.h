#pragma once

#include "core/common.h"
#include "core/map.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include "backend/optimizer.h"
#include <opencv2/features2d.hpp>
#include <mutex>
#include <thread>
#include <deque>
#include <condition_variable>
#include <vector>

#ifdef USE_DBOW2
#include <DBoW2/DBoW2.h>
#endif

namespace svslam {

#ifdef USE_DBOW2
// Define Vocabulary type for ORB
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbDatabase;
#endif

class LoopClosing {
public:
    using Ptr = std::shared_ptr<LoopClosing>;

    LoopClosing(Map::Ptr map, const std::string& vocab_path);

    // Main loop
    void run();

    // Input
    void insertKeyframe(Keyframe::Ptr kf);

    void requestStop();

    bool isEnabled() const { return enabled_; }

    // Set to true when metric depth is available (disables scale correction in loop closing)
    void setMetricDepth(bool metric) { has_metric_depth_ = metric; }

private:
    struct LoopConstraint {
        Keyframe::Ptr from;
        Keyframe::Ptr to;
        Sim3 relative_pose;
    };

    bool checkNewKeyframes();
    void processNewKeyframe();
    bool detectLoop();
    bool computeSim3();
    void correctLoop();
    std::vector<cv::DMatch> matchLoopCandidate() const;
    void fuseLoopLandmarks();
    void mergeLandmarks(const Landmark::Ptr& target, const Landmark::Ptr& source);

    Map::Ptr map_;
    std::string vocab_path_;
    bool enabled_ = false;

#ifdef USE_DBOW2
    std::shared_ptr<OrbVocabulary> vocab_;
    std::shared_ptr<OrbDatabase> db_;
    DBoW2::BowVector current_bow_vec_;
    DBoW2::FeatureVector current_feat_vec_;
#endif

    // Queue
    std::deque<Keyframe::Ptr> new_keyframes_;
    std::mutex mutex_new_keyframes_;
    std::condition_variable cv_new_keyframes_;

    Keyframe::Ptr current_processed_kf_;
    Keyframe::Ptr loop_candidate_kf_;
    Sim3 corrected_sim3_;
    std::vector<cv::DMatch> verified_loop_matches_;
    std::vector<LoopConstraint> loop_constraints_;

    std::vector<Keyframe::Ptr> db_keyframes_;
    int min_loop_interval_kf_ = 30;
    int max_loop_candidates_ = 4;
    double min_loop_score_ = 0.01;
    int min_loop_inliers_ = 30;
    int correction_window_size_ = 30;
    int loop_cooldown_kf_ = 120;
    int sim3_ransac_iterations_ = 200;
    double max_sim3_residual_ = 0.25;
    double min_sim3_scale_ = 0.7;
    double max_sim3_scale_ = 1.4;

    bool has_metric_depth_ = false;
    bool has_successful_loop_ = false;
    unsigned long last_successful_loop_kf_id_ = 0;
    bool stop_requested_ = false;
};

}
