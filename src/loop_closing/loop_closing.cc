#include "loop_closing/loop_closing.h"
#include <iostream>
#include <filesystem>

namespace svslam {

LoopClosing::LoopClosing(Map::Ptr map, const std::string& vocab_path)
    : map_(map), vocab_path_(vocab_path) {

#ifdef USE_DBOW2
    vocab_ = std::make_shared<OrbVocabulary>();

    // Check if vocab file exists
    if (!vocab_path_.empty() && std::filesystem::exists(vocab_path_)) {
        std::cout << "LoopClosing: Loading vocabulary from " << vocab_path_ << " ..." << std::endl;

        // Check file extension
        bool loaded = false;
        std::string ext = vocab_path_.substr(vocab_path_.find_last_of('.') + 1);

        try {
            if (ext == "yml" || ext == "yaml" || ext == "xml") {
                // OpenCV FileStorage format
                vocab_->load(vocab_path_);
                loaded = !vocab_->empty();
            } else if (ext == "txt") {
                // ORB-SLAM text format - not fully supported
                std::cout << "LoopClosing: Text format vocabulary detected." << std::endl;
                std::cout << "LoopClosing: Please convert to YAML format using DBoW2 tools." << std::endl;
                std::cout << "LoopClosing: Loop closing will be disabled for now." << std::endl;
                loaded = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "LoopClosing: Exception loading vocabulary: " << e.what() << std::endl;
        }

        if (loaded) {
            std::cout << "LoopClosing: Vocabulary loaded. " << vocab_->size() << " words." << std::endl;
            db_ = std::make_shared<OrbDatabase>(*vocab_, false, 0);
            enabled_ = true;
        } else {
            std::cout << "LoopClosing: Could not load vocabulary. Using online vocabulary building." << std::endl;
            // Enable online mode - we'll build vocabulary from keyframes
            db_ = nullptr;
            enabled_ = true; // Enable simplified loop detection
        }
    } else {
        std::cout << "LoopClosing: No vocab file specified. Using descriptor matching for loop detection." << std::endl;
        enabled_ = true; // Enable simplified loop detection without BoW
    }
#else
    std::cout << "LoopClosing: DBoW2 not available. Loop closing disabled." << std::endl;
#endif
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

    if (!enabled_) return;

    // Add to keyframe list
    db_keyframes_.push_back(current_processed_kf_);

#ifdef USE_DBOW2
    // Use BoW if vocabulary is available
    if (vocab_ && !vocab_->empty() && db_) {
        // Convert descriptors to vector format for DBoW2
        std::vector<cv::Mat> descriptors;
        cv::Mat desc = current_processed_kf_->descriptors_;
        for (int i = 0; i < desc.rows; ++i) {
            descriptors.push_back(desc.row(i));
        }

        // Transform to BoW
        vocab_->transform(descriptors, current_bow_vec_, current_feat_vec_, 4);

        // Add to database
        db_->add(current_bow_vec_, current_feat_vec_);
    }
#endif

    // Detect loop (works with or without BoW)
    if (detectLoop()) {
        std::cout << "LoopClosing: Loop detected between KF " << current_processed_kf_->id_
                  << " and KF " << loop_candidate_kf_->id_ << std::endl;

        if (computeSim3()) {
            correctLoop();
        }
    }
}

bool LoopClosing::detectLoop() {
    if (db_keyframes_.size() < static_cast<size_t>(min_loop_interval_kf_)) {
        return false;
    }

#ifdef USE_DBOW2
    // Use BoW-based detection if vocabulary is available
    if (db_ && vocab_ && !vocab_->empty()) {
        DBoW2::QueryResults results;
        db_->query(current_bow_vec_, results, max_loop_candidates_ + 1);

        for (const auto& r : results) {
            const int db_idx = r.Id;
            if (db_idx < 0 || db_idx >= static_cast<int>(db_keyframes_.size())) continue;

            auto cand_kf = db_keyframes_[db_idx];
            if (!cand_kf) continue;

            const long diff = static_cast<long>(current_processed_kf_->id_) - static_cast<long>(cand_kf->id_);
            if (std::abs(diff) < min_loop_interval_kf_) continue;

            if (r.Score < min_loop_score_) continue;

            std::cout << "LoopClosing: Loop candidate (BoW). cur_kf=" << current_processed_kf_->id_
                      << " cand_kf=" << cand_kf->id_ << " score=" << r.Score << std::endl;

            loop_candidate_kf_ = cand_kf;
            return true;
        }
        return false;
    }
#endif

    // Fallback: descriptor matching based loop detection
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    const int min_matches_for_loop = 50;

    for (size_t i = 0; i + min_loop_interval_kf_ < db_keyframes_.size(); ++i) {
        auto cand_kf = db_keyframes_[i];
        if (!cand_kf) continue;

        const long diff = static_cast<long>(current_processed_kf_->id_) - static_cast<long>(cand_kf->id_);
        if (std::abs(diff) < min_loop_interval_kf_) continue;

        // Match descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(current_processed_kf_->descriptors_, cand_kf->descriptors_, knn_matches, 2);

        // Apply ratio test
        int good_matches = 0;
        for (const auto& m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < 0.7f * m[1].distance) {
                good_matches++;
            }
        }

        if (good_matches >= min_matches_for_loop) {
            std::cout << "LoopClosing: Loop candidate (desc). cur_kf=" << current_processed_kf_->id_
                      << " cand_kf=" << cand_kf->id_ << " matches=" << good_matches << std::endl;
            loop_candidate_kf_ = cand_kf;
            return true;
        }
    }

    return false;
}

bool LoopClosing::computeSim3() {
    // TODO: Compute similarity transform between current and loop candidate
    // This involves:
    // 1. Find correspondences using BoW
    // 2. Compute Sim3 using RANSAC
    // 3. Verify with enough inliers

    return false; // Stub for now
}

void LoopClosing::correctLoop() {
    // TODO: Correct the loop
    // 1. Fuse duplicate map points
    // 2. Optimize pose graph
    // 3. Update map

    std::cout << "LoopClosing: Loop correction (stub)" << std::endl;
}

}
