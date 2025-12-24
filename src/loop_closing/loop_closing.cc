#include "loop_closing/loop_closing.h"
#include <iostream>

namespace svslam {

LoopClosing::LoopClosing(Map::Ptr map, const std::string& vocab_path) 
    : map_(map), vocab_path_(vocab_path) {
    
    vocab_ = std::make_shared<OrbVocabulary>();
    
    std::cout << "LoopClosing: Loading vocabulary from " << vocab_path_ << " ..." << std::endl;
    bool loaded = false;
    try {
        // loaded = vocab_->loadFromTextFile(vocab_path_); // load method name might differ in DBoW2 versions
        // Common DBoW2 load method: load
        // DBoW2::Vocabulary::load(filename)
        // Let's assume standard interface
        // Note: DBoW2 interfaces can be inconsistent across forks. 
        // dorian3d/DBoW2 uses `load` for files.
        // It detects extension.
        
        if (!vocab_path_.empty()) {
            vocab_->load(vocab_path_);
            loaded = true;
        }
    } catch (const std::exception& e) {
        std::cerr << "LoopClosing: Failed to load vocabulary: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "LoopClosing: Failed to load vocabulary: unknown exception" << std::endl;
    }
    
    if (loaded) {
        std::cout << "LoopClosing: Vocabulary loaded. " << vocab_->size() << " words." << std::endl;
        db_ = std::make_shared<OrbDatabase>(*vocab_, false, 0); // Direct index, inverse index
    } else {
        std::cout << "LoopClosing: Vocabulary not loaded (or empty path). Loop closing disabled." << std::endl;
    }
}

void LoopClosing::insertKeyframe(Keyframe::Ptr kf) {
    std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
    new_keyframes_.push_back(kf);
    cv_new_keyframes_.notify_one();
}

void LoopClosing::requestStop() {
    stop_requested_ = true;
    cv_new_keyframes_.notify_one();
}

void LoopClosing::run() {
    std::cout << "LoopClosing thread started." << std::endl;
    
    while (!stop_requested_) {
        {
            std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
            if (new_keyframes_.empty()) {
                cv_new_keyframes_.wait(lock);
            }
            if (stop_requested_) break;
        }
        
        if (checkNewKeyframes()) {
            processNewKeyframe();
        }
    }
    
    std::cout << "LoopClosing thread stopped." << std::endl;
}

bool LoopClosing::checkNewKeyframes() {
    std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
    return !new_keyframes_.empty();
}

void LoopClosing::processNewKeyframe() {
    {
        std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
        current_processed_kf_ = new_keyframes_.front();
        new_keyframes_.pop_front();
    }
    
    // Convert to BoW
    if (vocab_ && !vocab_->empty() && db_) {
        std::vector<cv::Mat> descriptors;
        // descriptors must be vector of Mat (one row per feature)
        // current_processed_kf_->descriptors_ is a single Mat (N rows)
        
        cv::Mat desc = current_processed_kf_->descriptors_;
        for (int i=0; i<desc.rows; ++i) {
            descriptors.push_back(desc.row(i));
        }
        
        DBoW2::BowVector bow_vec;
        DBoW2::FeatureVector feat_vec;
        
        vocab_->transform(descriptors, bow_vec, feat_vec, 4); // 4 levels
        
        // Add to database (store Keyframe index correspondence)
        db_->add(bow_vec, feat_vec);
        db_keyframes_.push_back(current_processed_kf_);

        // Loop candidate search
        detectLoop(bow_vec);
    }
}

bool LoopClosing::detectLoop(const DBoW2::BowVector& bow_vec) {
    if (!db_ || db_keyframes_.empty() || !current_processed_kf_) return false;

    DBoW2::QueryResults results;
    db_->query(bow_vec, results, max_loop_candidates_);

    bool found = false;
    for (const auto& r : results) {
        const int db_idx = r.Id;
        if (db_idx < 0 || db_idx >= static_cast<int>(db_keyframes_.size())) continue;

        auto cand_kf = db_keyframes_[db_idx];
        if (!cand_kf) continue;

        // Filter out recent keyframes
        const long diff = static_cast<long>(current_processed_kf_->id_) - static_cast<long>(cand_kf->id_);
        if (std::abs(diff) < min_loop_interval_kf_) continue;

        if (r.Score < min_loop_score_) continue;

        std::cout << "LoopClosing: Loop candidate found. cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << cand_kf->id_ << " score=" << r.Score << std::endl;
        found = true;
    }

    return found;
}

bool LoopClosing::computeSim3() {
    return false;
}

void LoopClosing::correctLoop() {
    
}

}
