#include "loop_closing/loop_closing.h"
#include <iostream>
#include <filesystem>
#include <set>
#include <array>
#include <limits>
#include <numeric>
#include <random>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

namespace {

std::vector<cv::DMatch> ratioMatches(const cv::Mat& query_desc, const cv::Mat& train_desc) {
    std::vector<cv::DMatch> good_matches;
    if (query_desc.empty() || train_desc.empty()) return good_matches;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(query_desc, train_desc, knn_matches, 2);

    for (const auto& m : knn_matches) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.7f * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    return good_matches;
}

struct Loop3dCorrespondence {
    cv::DMatch match;
    svslam::Vec3 current_pos;
    svslam::Vec3 candidate_pos;
};

bool isFinite(const svslam::Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool sampleMinimalSet(
    const std::vector<Loop3dCorrespondence>& correspondences,
    std::mt19937& rng,
    std::array<int, 3>& indices) {
    if (correspondences.size() < indices.size()) return false;

    std::uniform_int_distribution<int> dist(0, static_cast<int>(correspondences.size()) - 1);
    for (int tries = 0; tries < 32; ++tries) {
        indices[0] = dist(rng);
        indices[1] = dist(rng);
        indices[2] = dist(rng);

        if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2]) {
            continue;
        }

        const auto& p0 = correspondences[indices[0]].current_pos;
        const auto& p1 = correspondences[indices[1]].current_pos;
        const auto& p2 = correspondences[indices[2]].current_pos;
        if (((p1 - p0).cross(p2 - p0)).norm() < 1e-4) {
            continue;
        }
        return true;
    }
    return false;
}

bool estimateSim3FromIndices(
    const std::vector<Loop3dCorrespondence>& correspondences,
    const std::vector<int>& indices,
    svslam::Sim3& sim3,
    double min_scale,
    double max_scale) {
    if (indices.size() < 3) return false;

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, indices.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst(3, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& corr = correspondences[indices[i]];
        src.col(i) = corr.current_pos;
        dst.col(i) = corr.candidate_pos;
    }

    const Eigen::Matrix4d T = Eigen::umeyama(src, dst, true);
    if (!T.allFinite()) return false;

    const Eigen::Matrix3d sR = T.topLeftCorner<3, 3>();
    const double scale =
        (sR.col(0).norm() + sR.col(1).norm() + sR.col(2).norm()) / 3.0;
    if (!std::isfinite(scale) || scale < min_scale || scale > max_scale) {
        return false;
    }

    const Eigen::Matrix3d R = sR / scale;
    if (!R.allFinite()) return false;

    Eigen::Quaterniond q(R);
    if (!std::isfinite(q.w()) || !std::isfinite(q.x()) ||
        !std::isfinite(q.y()) || !std::isfinite(q.z())) {
        return false;
    }
    q.normalize();

    const svslam::Vec3 t = T.topRightCorner<3, 1>();
    if (!isFinite(t)) return false;

    sim3 = svslam::Sim3(scale, q, t);
    return true;
}

std::vector<int> computeSim3Inliers(
    const std::vector<Loop3dCorrespondence>& correspondences,
    const svslam::Sim3& sim3,
    double max_residual) {
    std::vector<int> inliers;
    inliers.reserve(correspondences.size());
    for (size_t i = 0; i < correspondences.size(); ++i) {
        const auto& corr = correspondences[i];
        const double residual = (sim3 * corr.current_pos - corr.candidate_pos).norm();
        if (residual < max_residual) {
            inliers.push_back(static_cast<int>(i));
        }
    }
    return inliers;
}

}  // namespace

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
                // ORB-SLAM text format
                std::cout << "LoopClosing: Loading ORB-SLAM text format vocabulary..." << std::endl;
                vocab_->loadFromTextFile(vocab_path_);
                loaded = !vocab_->empty();
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
            has_successful_loop_ = true;
            last_successful_loop_kf_id_ = current_processed_kf_->id_;
        }
    }
}

bool LoopClosing::detectLoop() {
    if (has_successful_loop_ &&
        current_processed_kf_ &&
        current_processed_kf_->id_ < last_successful_loop_kf_id_ + static_cast<unsigned long>(loop_cooldown_kf_)) {
        return false;
    }

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
    const int min_matches_for_loop = 50;

    for (size_t i = 0; i + min_loop_interval_kf_ < db_keyframes_.size(); ++i) {
        auto cand_kf = db_keyframes_[i];
        if (!cand_kf) continue;

        const long diff = static_cast<long>(current_processed_kf_->id_) - static_cast<long>(cand_kf->id_);
        if (std::abs(diff) < min_loop_interval_kf_) continue;

        const auto matches = ratioMatches(current_processed_kf_->descriptors_, cand_kf->descriptors_);
        const int good_matches = static_cast<int>(matches.size());

        if (good_matches >= min_matches_for_loop) {
            std::cout << "LoopClosing: Loop candidate (desc). cur_kf=" << current_processed_kf_->id_
                      << " cand_kf=" << cand_kf->id_ << " matches=" << good_matches << std::endl;
            loop_candidate_kf_ = cand_kf;
            return true;
        }
    }

    return false;
}

std::vector<cv::DMatch> LoopClosing::matchLoopCandidate() const {
    if (!current_processed_kf_ || !loop_candidate_kf_) return {};
    return ratioMatches(current_processed_kf_->descriptors_, loop_candidate_kf_->descriptors_);
}

bool LoopClosing::computeSim3() {
    verified_loop_matches_.clear();
    if (!current_processed_kf_ || !loop_candidate_kf_) return false;

    const auto matches = matchLoopCandidate();
    if (matches.size() < static_cast<size_t>(min_loop_inliers_)) {
        return false;
    }

    std::vector<Loop3dCorrespondence> correspondences;
    correspondences.reserve(matches.size());

    for (const auto& match : matches) {
        if (match.queryIdx < 0 || match.trainIdx < 0) continue;
        if (match.queryIdx >= static_cast<int>(current_processed_kf_->landmarks_.size())) continue;
        if (match.trainIdx >= static_cast<int>(loop_candidate_kf_->landmarks_.size())) continue;

        auto current_lm = current_processed_kf_->landmarks_[match.queryIdx];
        auto candidate_lm = loop_candidate_kf_->landmarks_[match.trainIdx];
        if (!current_lm || !candidate_lm) continue;
        if (current_lm->isBad() || candidate_lm->isBad()) continue;

        const Vec3 current_pos = current_lm->getPos();
        const Vec3 candidate_pos = candidate_lm->getPos();
        if (!isFinite(current_pos) || !isFinite(candidate_pos)) continue;

        correspondences.push_back({match, current_pos, candidate_pos});
    }

    if (correspondences.size() < static_cast<size_t>(min_loop_inliers_)) {
        return false;
    }

    std::mt19937 rng(static_cast<uint32_t>(
        current_processed_kf_->id_ * 73856093ULL ^
        loop_candidate_kf_->id_ * 19349663ULL));

    Sim3 best_sim3;
    std::vector<int> best_inliers;
    best_inliers.reserve(correspondences.size());
    std::array<int, 3> sample_indices{};

    // Tighter scale bounds when metric depth is available
    const double eff_min_scale = has_metric_depth_ ? 0.85 : min_sim3_scale_;
    const double eff_max_scale = has_metric_depth_ ? 1.15 : max_sim3_scale_;

    for (int iter = 0; iter < sim3_ransac_iterations_; ++iter) {
        if (!sampleMinimalSet(correspondences, rng, sample_indices)) {
            break;
        }

        std::vector<int> minimal_indices(sample_indices.begin(), sample_indices.end());
        Sim3 candidate_sim3;
        if (!estimateSim3FromIndices(
                correspondences,
                minimal_indices,
                candidate_sim3,
                eff_min_scale,
                eff_max_scale)) {
            continue;
        }

        auto inliers = computeSim3Inliers(correspondences, candidate_sim3, max_sim3_residual_);
        if (inliers.size() > best_inliers.size()) {
            best_inliers = std::move(inliers);
            best_sim3 = candidate_sim3;
        }
    }

    if (best_inliers.size() < static_cast<size_t>(min_loop_inliers_)) {
        return false;
    }

    Sim3 refined_sim3;
    if (!estimateSim3FromIndices(
            correspondences,
            best_inliers,
            refined_sim3,
            eff_min_scale,
            eff_max_scale)) {
        return false;
    }

    best_inliers = computeSim3Inliers(correspondences, refined_sim3, max_sim3_residual_);
    if (best_inliers.size() < static_cast<size_t>(min_loop_inliers_)) {
        return false;
    }

    corrected_sim3_ = refined_sim3;
    verified_loop_matches_.reserve(best_inliers.size());
    for (const int idx : best_inliers) {
        if (idx < 0 || idx >= static_cast<int>(correspondences.size())) continue;
        verified_loop_matches_.push_back(correspondences[idx].match);
    }

    std::cout << "LoopClosing: Geometric verification succeeded with "
              << verified_loop_matches_.size() << " inliers"
              << " | sim3_scale=" << corrected_sim3_.scale() << std::endl;
    return true;
}

void LoopClosing::mergeLandmarks(const Landmark::Ptr& target, const Landmark::Ptr& source) {
    if (!target || !source || target == source || source->isBad()) return;

    // Deadlock-free double lock by ID ordering
    auto& first_mtx = (target->id_ < source->id_) ? target->mutex_ : source->mutex_;
    auto& second_mtx = (target->id_ < source->id_) ? source->mutex_ : target->mutex_;
    std::lock_guard<std::mutex> lock1(first_mtx);
    std::lock_guard<std::mutex> lock2(second_mtx);

    std::vector<std::pair<Keyframe::Ptr, size_t>> source_observations;
    {
        for (const auto& obs : source->observations_) {
            auto kf = obs.first.lock();
            if (kf) {
                source_observations.push_back({kf, obs.second});
            }
        }
        source->observations_.clear();
    }

    for (const auto& obs : source_observations) {
        auto kf = obs.first;
        const size_t idx = obs.second;
        if (!kf || idx >= kf->landmarks_.size()) continue;

        std::unique_lock<std::mutex> kf_lock(kf->mutex_);
        if (!kf->landmarks_[idx] || kf->landmarks_[idx] == source) {
            kf->landmarks_[idx] = target;
            target->addObservation(kf, idx);
        }
    }

    source->setBad();
    map_->removeLandmark(source);
}

void LoopClosing::fuseLoopLandmarks() {
    for (const auto& match : verified_loop_matches_) {
        if (match.queryIdx < 0 || match.trainIdx < 0) continue;
        if (match.queryIdx >= static_cast<int>(current_processed_kf_->landmarks_.size())) continue;
        if (match.trainIdx >= static_cast<int>(loop_candidate_kf_->landmarks_.size())) continue;

        auto current_lm = current_processed_kf_->landmarks_[match.queryIdx];
        auto candidate_lm = loop_candidate_kf_->landmarks_[match.trainIdx];

        if (candidate_lm && candidate_lm->isBad()) candidate_lm.reset();
        if (current_lm && current_lm->isBad()) current_lm.reset();

        if (!candidate_lm && !current_lm) continue;

        if (!current_lm && candidate_lm) {
            current_processed_kf_->landmarks_[match.queryIdx] = candidate_lm;
            candidate_lm->addObservation(current_processed_kf_, match.queryIdx);
            continue;
        }

        if (current_lm && candidate_lm && current_lm != candidate_lm) {
            mergeLandmarks(candidate_lm, current_lm);
        }
    }
}

void LoopClosing::correctLoop() {
    if (!current_processed_kf_ || !loop_candidate_kf_) return;

    const Sim3 current_pose_sim3(
        1.0,
        current_processed_kf_->T_cw_.unit_quaternion(),
        current_processed_kf_->T_cw_.translation());
    const Sim3 candidate_pose_sim3(
        1.0,
        loop_candidate_kf_->T_cw_.unit_quaternion(),
        loop_candidate_kf_->T_cw_.translation());
    const Sim3 corrected_current_pose_sim3 = current_pose_sim3 * corrected_sim3_.inverse();

    LoopConstraint constraint;
    constraint.from = current_processed_kf_;
    constraint.to = loop_candidate_kf_;
    constraint.relative_pose = candidate_pose_sim3 * corrected_current_pose_sim3.inverse();
    loop_constraints_.push_back(constraint);

    std::vector<Optimizer::PoseGraphEdge> optimizer_edges;
    optimizer_edges.reserve(loop_constraints_.size());
    for (const auto& loop_constraint : loop_constraints_) {
        Optimizer::PoseGraphEdge edge;
        edge.from = loop_constraint.from;
        edge.to = loop_constraint.to;
        edge.relative_pose = loop_constraint.relative_pose;
        edge.translation_weight = 10.0;
        edge.rotation_weight = 10.0;
        // Lock scale when metric depth is available
        edge.scale_weight = has_metric_depth_ ? 1000.0 : 15.0;
        optimizer_edges.push_back(edge);
    }

    Optimizer::poseGraphOptimization(map_, optimizer_edges, 60);

    // Protect structural modifications (landmark merge, covisibility graph update)
    map_->loop_correcting_.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    fuseLoopLandmarks();

    const auto& all_keyframes = map_->getAllKeyframes();
    for (const auto& kv : all_keyframes) {
        if (kv.second) kv.second->updateConnections();
    }

    map_->loop_correcting_.store(false);

    std::cout << "LoopClosing: Applied global pose graph with "
              << optimizer_edges.size() << " loop constraints and global BA." << std::endl;
}

}
