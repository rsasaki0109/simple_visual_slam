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

std::vector<cv::DMatch> ratioMatches(
    const cv::Mat& query_desc,
    const cv::Mat& train_desc,
    float ratio_threshold = 0.7f) {
    std::vector<cv::DMatch> good_matches;
    if (query_desc.empty() || train_desc.empty()) return good_matches;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(query_desc, train_desc, knn_matches, 2);

    for (const auto& m : knn_matches) {
        if (m.size() < 2) continue;
        if (m[0].distance < ratio_threshold * m[1].distance) {
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

struct LoopConstraintConsistency {
    bool valid = false;
    double translation_error = std::numeric_limits<double>::infinity();
    double scale_error = std::numeric_limits<double>::infinity();
};

bool isFinite(const svslam::Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool getMatchedLandmarks(
    const svslam::Keyframe::Ptr& current_kf,
    const svslam::Keyframe::Ptr& candidate_kf,
    const cv::DMatch& match,
    svslam::Landmark::Ptr& current_lm,
    svslam::Landmark::Ptr& candidate_lm) {
    if (!current_kf || !candidate_kf) return false;
    if (match.queryIdx < 0 || match.trainIdx < 0) return false;

    if (current_kf == candidate_kf) {
        std::lock_guard<std::mutex> lock(current_kf->mutex_);
        if (match.queryIdx >= static_cast<int>(current_kf->landmarks_.size()) ||
            match.trainIdx >= static_cast<int>(candidate_kf->landmarks_.size())) {
            return false;
        }
        current_lm = current_kf->landmarks_[match.queryIdx];
        candidate_lm = candidate_kf->landmarks_[match.trainIdx];
        return true;
    }

    std::scoped_lock lock(current_kf->mutex_, candidate_kf->mutex_);
    if (match.queryIdx >= static_cast<int>(current_kf->landmarks_.size()) ||
        match.trainIdx >= static_cast<int>(candidate_kf->landmarks_.size())) {
        return false;
    }
    current_lm = current_kf->landmarks_[match.queryIdx];
    candidate_lm = candidate_kf->landmarks_[match.trainIdx];
    return true;
}

int countLoop3dCorrespondences(
    const svslam::Keyframe::Ptr& current_kf,
    const svslam::Keyframe::Ptr& candidate_kf,
    const std::vector<cv::DMatch>& matches) {
    int usable_correspondences = 0;
    for (const auto& match : matches) {
        svslam::Landmark::Ptr current_lm;
        svslam::Landmark::Ptr candidate_lm;
        if (!getMatchedLandmarks(current_kf, candidate_kf, match, current_lm, candidate_lm)) {
            continue;
        }
        if (!current_lm || !candidate_lm) continue;
        if (current_lm->isBad() || candidate_lm->isBad()) continue;

        const auto current_pos = current_lm->getPos();
        const auto candidate_pos = candidate_lm->getPos();
        if (!isFinite(current_pos) || !isFinite(candidate_pos)) continue;

        ++usable_correspondences;
    }
    return usable_correspondences;
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
    double max_scale,
    bool estimate_scale = true) {
    if (indices.size() < 3) return false;

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, indices.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst(3, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& corr = correspondences[indices[i]];
        src.col(i) = corr.current_pos;
        dst.col(i) = corr.candidate_pos;
    }

    const Eigen::Matrix4d T = Eigen::umeyama(src, dst, estimate_scale);
    if (!T.allFinite()) return false;

    const Eigen::Matrix3d sR = T.topLeftCorner<3, 3>();
    const double scale = estimate_scale
        ? (sR.col(0).norm() + sR.col(1).norm() + sR.col(2).norm()) / 3.0
        : 1.0;
    if (!std::isfinite(scale) || scale < min_scale || scale > max_scale) {
        return false;
    }

    const Eigen::Matrix3d R = estimate_scale ? (sR / scale) : sR;
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

LoopConstraintConsistency evaluateLoopConstraintConsistency(
    const svslam::Keyframe::Ptr& from,
    const svslam::Keyframe::Ptr& to,
    const svslam::Sim3& reference_relative_pose) {
    LoopConstraintConsistency consistency;
    if (!from || !to) return consistency;

    Eigen::Quaterniond q_from = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond q_to = Eigen::Quaterniond::Identity();
    svslam::Vec3 t_from = svslam::Vec3::Zero();
    svslam::Vec3 t_to = svslam::Vec3::Zero();
    if (from == to) {
        std::lock_guard<std::mutex> lock(from->mutex_);
        q_from = from->T_cw_.unit_quaternion();
        q_to = to->T_cw_.unit_quaternion();
        t_from = from->T_cw_.translation();
        t_to = to->T_cw_.translation();
    } else {
        std::scoped_lock lock(from->mutex_, to->mutex_);
        q_from = from->T_cw_.unit_quaternion();
        q_to = to->T_cw_.unit_quaternion();
        t_from = from->T_cw_.translation();
        t_to = to->T_cw_.translation();
    }

    const svslam::Sim3 current_relative_pose =
        svslam::Sim3(1.0, q_to, t_to) *
        svslam::Sim3(1.0, q_from, t_from).inverse();
    const svslam::Sim3 delta = current_relative_pose * reference_relative_pose.inverse();

    consistency.valid = true;
    consistency.translation_error = delta.translation().norm();
    consistency.scale_error = std::abs(delta.scale() - 1.0);
    return consistency;
}

double computeLoopConstraintReuseDecay(const LoopConstraintConsistency& consistency) {
    if (!consistency.valid) return 1.0;
    return svslam::loop_closing_internal::computeStaleLoopEdgeDecay(
        consistency.translation_error, consistency.scale_error);
}

}  // namespace

namespace svslam {

namespace loop_closing_internal {

bool isFinalSim3ScaleAcceptable(double scale,
                                bool has_metric_depth,
                                double metric_scale_tolerance) {
    if (!std::isfinite(scale)) {
        return false;
    }
    return !has_metric_depth || std::abs(scale - 1.0) <= metric_scale_tolerance;
}

LoopConstraintWeighting computeLoopConstraintWeighting(int inlier_count,
                                                       double inlier_ratio,
                                                       bool has_metric_depth) {
    LoopConstraintWeighting weighting;
    const double support_norm =
        std::clamp(static_cast<double>(inlier_count) / 40.0, 0.25, 1.0);
    const double ratio_norm =
        std::clamp(inlier_ratio / 0.60, 0.50, 1.0);
    weighting.confidence = std::sqrt(support_norm) * ratio_norm;
    weighting.translation_weight = has_metric_depth
        ? (3.0 + 4.0 * weighting.confidence)
        : 10.0;
    weighting.rotation_weight = weighting.translation_weight;
    weighting.scale_weight = has_metric_depth ? 1000.0 : 15.0;
    return weighting;
}

double computeStaleLoopEdgeDecay(double translation_error, double scale_error) {
    const double translation_decay = translation_error <= 0.15
        ? 1.0
        : std::clamp(0.15 / translation_error, 0.35, 1.0);
    const double scale_decay = scale_error <= 0.02
        ? 1.0
        : std::clamp(0.02 / scale_error, 0.35, 1.0);
    return std::min(translation_decay, scale_decay);
}

double computeLoopConstraintOverlapDecay(unsigned long newest_from_id,
                                         unsigned long newest_to_id,
                                         unsigned long existing_from_id,
                                         unsigned long existing_to_id,
                                         unsigned long overlap_window_kf) {
    if (overlap_window_kf == 0) {
        return 1.0;
    }

    const auto endpoint_overlaps = [overlap_window_kf](unsigned long lhs, unsigned long rhs) {
        const auto diff = (lhs > rhs) ? (lhs - rhs) : (rhs - lhs);
        return diff < overlap_window_kf;
    };

    const bool from_overlaps = endpoint_overlaps(newest_from_id, existing_from_id);
    const bool to_overlaps = endpoint_overlaps(newest_to_id, existing_to_id);
    if (from_overlaps && to_overlaps) {
        return 0.35;
    }
    if (from_overlaps || to_overlaps) {
        return 0.60;
    }
    return 1.0;
}

void mergeLandmarks(Map::Ptr map,
                    const Landmark::Ptr& target,
                    const Landmark::Ptr& source) {
    if (!map || !target || !source || target == source || source->isBad()) {
        return;
    }

    auto& first_mutex = (target->id_ < source->id_) ? target->mutex_ : source->mutex_;
    auto& second_mutex = (target->id_ < source->id_) ? source->mutex_ : target->mutex_;
    std::lock_guard<std::mutex> first_lock(first_mutex);
    std::lock_guard<std::mutex> second_lock(second_mutex);

    std::vector<std::pair<Keyframe::Ptr, size_t>> source_observations;
    source_observations.reserve(source->observations_.size());
    for (const auto& observation : source->observations_) {
        auto keyframe = observation.first.lock();
        if (keyframe) {
            source_observations.push_back({keyframe, observation.second});
        }
    }
    source->observations_.clear();

    for (const auto& observation : source_observations) {
        const auto& keyframe = observation.first;
        const size_t index = observation.second;
        if (!keyframe || index >= keyframe->landmarks_.size()) {
            continue;
        }

        std::unique_lock<std::mutex> keyframe_lock(keyframe->mutex_);
        if (!keyframe->landmarks_[index] || keyframe->landmarks_[index] == source) {
            keyframe->landmarks_[index] = target;
            target->observations_[keyframe] = index;
        }
    }

    source->setBad();
    map->removeLandmark(source);
}

}  // namespace loop_closing_internal

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
    const int min_matches_for_loop = 50;

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

        Keyframe::Ptr best_candidate;
        int best_good_matches = 0;
        int best_usable_correspondences = 0;
        double best_score = 0.0;
        for (const auto& r : results) {
            const int db_idx = r.Id;
            if (db_idx < 0 || db_idx >= static_cast<int>(db_keyframes_.size())) continue;

            auto cand_kf = db_keyframes_[db_idx];
            if (!cand_kf) continue;

            const long diff = static_cast<long>(current_processed_kf_->id_) - static_cast<long>(cand_kf->id_);
            if (std::abs(diff) < min_loop_interval_kf_) continue;

            if (r.Score < min_loop_score_) continue;

            const auto matches =
                ratioMatches(current_processed_kf_->descriptors_, cand_kf->descriptors_, 0.8f);
            const int good_matches = static_cast<int>(matches.size());
            const int usable_correspondences =
                countLoop3dCorrespondences(current_processed_kf_, cand_kf, matches);

            std::cout << "LoopClosing: Loop candidate (BoW). cur_kf=" << current_processed_kf_->id_
                      << " cand_kf=" << cand_kf->id_ << " score=" << r.Score
                      << " desc_matches=" << good_matches
                      << " corr3d=" << usable_correspondences << std::endl;

            if (good_matches < min_matches_for_loop ||
                usable_correspondences < min_loop_inliers_) {
                continue;
            }

            if (usable_correspondences > best_usable_correspondences ||
                (usable_correspondences == best_usable_correspondences &&
                 good_matches > best_good_matches) ||
                (usable_correspondences == best_usable_correspondences &&
                 good_matches == best_good_matches &&
                 r.Score > best_score)) {
                best_candidate = cand_kf;
                best_usable_correspondences = usable_correspondences;
                best_good_matches = good_matches;
                best_score = r.Score;
            }
        }

        if (best_candidate) {
            loop_candidate_kf_ = best_candidate;
            return true;
        }

        std::cout << "LoopClosing: BoW candidates below descriptor-match threshold, "
                  << "falling back to descriptor scan." << std::endl;
    }
#endif

    // Fallback: descriptor matching based loop detection
    Keyframe::Ptr best_candidate;
    int best_good_matches = 0;
    int best_usable_correspondences = 0;

    for (size_t i = 0; i + min_loop_interval_kf_ < db_keyframes_.size(); ++i) {
        auto cand_kf = db_keyframes_[i];
        if (!cand_kf) continue;

        const long diff = static_cast<long>(current_processed_kf_->id_) - static_cast<long>(cand_kf->id_);
        if (std::abs(diff) < min_loop_interval_kf_) continue;

        const auto matches = ratioMatches(current_processed_kf_->descriptors_, cand_kf->descriptors_, 0.8f);
        const int good_matches = static_cast<int>(matches.size());
        if (good_matches < min_matches_for_loop) continue;

        const int usable_correspondences =
            countLoop3dCorrespondences(current_processed_kf_, cand_kf, matches);
        if (usable_correspondences < min_loop_inliers_) continue;

        if (usable_correspondences > best_usable_correspondences ||
            (usable_correspondences == best_usable_correspondences &&
             good_matches > best_good_matches)) {
            best_candidate = cand_kf;
            best_usable_correspondences = usable_correspondences;
            best_good_matches = good_matches;
        }
    }

    if (best_candidate) {
        std::cout << "LoopClosing: Loop candidate (desc). cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << best_candidate->id_ << " matches=" << best_good_matches
                  << " corr3d=" << best_usable_correspondences << std::endl;
        loop_candidate_kf_ = best_candidate;
        return true;
    }

    return false;
}

std::vector<cv::DMatch> LoopClosing::matchLoopCandidate() const {
    if (!current_processed_kf_ || !loop_candidate_kf_) return {};
    return ratioMatches(current_processed_kf_->descriptors_, loop_candidate_kf_->descriptors_, 0.8f);
}

bool LoopClosing::computeSim3() {
    verified_loop_matches_.clear();
    corrected_loop_inlier_count_ = 0;
    corrected_loop_inlier_ratio_ = 0.0;
    if (!current_processed_kf_ || !loop_candidate_kf_) return false;

    const auto matches = matchLoopCandidate();
    if (matches.size() < static_cast<size_t>(min_loop_inliers_)) {
        std::cout << "LoopClosing: computeSim3 rejected candidate cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << loop_candidate_kf_->id_
                  << " reason=insufficient_descriptor_matches matches=" << matches.size()
                  << " min_required=" << min_loop_inliers_ << std::endl;
        return false;
    }

    std::vector<Loop3dCorrespondence> correspondences;
    correspondences.reserve(matches.size());

    for (const auto& match : matches) {
        Landmark::Ptr current_lm;
        Landmark::Ptr candidate_lm;
        if (!getMatchedLandmarks(current_processed_kf_, loop_candidate_kf_, match, current_lm, candidate_lm)) {
            continue;
        }
        if (!current_lm || !candidate_lm) continue;
        if (current_lm->isBad() || candidate_lm->isBad()) continue;

        const Vec3 current_pos = current_lm->getPos();
        const Vec3 candidate_pos = candidate_lm->getPos();
        if (!isFinite(current_pos) || !isFinite(candidate_pos)) continue;

        correspondences.push_back({match, current_pos, candidate_pos});
    }

    if (correspondences.size() < static_cast<size_t>(min_loop_inliers_)) {
        std::cout << "LoopClosing: computeSim3 rejected candidate cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << loop_candidate_kf_->id_
                  << " reason=insufficient_3d_correspondences correspondences=" << correspondences.size()
                  << " matches=" << matches.size()
                  << " min_required=" << min_loop_inliers_ << std::endl;
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
    const double eff_max_residual = max_sim3_residual_;
    const double diagnostic_relaxed_residual = has_metric_depth_ ? 0.35 : (eff_max_residual + 0.10);
    const int refinement_seed_min_inliers = has_metric_depth_ ? 15 : min_loop_inliers_;
    const int final_min_inliers = has_metric_depth_ ? 22 : min_loop_inliers_;
    const double final_min_inlier_ratio = has_metric_depth_ ? 0.38 : 0.0;
    const double metric_scale_tolerance = 0.05;
    const bool estimate_scale = !has_metric_depth_;
    bool have_best_sim3 = false;

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
                eff_max_scale,
                estimate_scale)) {
            continue;
        }

        auto inliers = computeSim3Inliers(correspondences, candidate_sim3, eff_max_residual);
        if (inliers.size() > best_inliers.size()) {
            best_inliers = std::move(inliers);
            best_sim3 = candidate_sim3;
            have_best_sim3 = true;
        }
    }

    if (best_inliers.size() < static_cast<size_t>(refinement_seed_min_inliers)) {
        size_t relaxed_inliers = best_inliers.size();
        if (have_best_sim3 && diagnostic_relaxed_residual > eff_max_residual) {
            relaxed_inliers =
                computeSim3Inliers(correspondences, best_sim3, diagnostic_relaxed_residual).size();
        }
        std::cout << "LoopClosing: computeSim3 rejected candidate cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << loop_candidate_kf_->id_
                  << " reason=insufficient_ransac_inliers correspondences=" << correspondences.size()
                  << " best_inliers=" << best_inliers.size()
                  << " refine_seed_min=" << refinement_seed_min_inliers
                  << " final_min=" << final_min_inliers
                  << " residual_thresh=" << eff_max_residual
                  << " relaxed_inliers=" << relaxed_inliers
                  << " relaxed_residual_thresh=" << diagnostic_relaxed_residual << std::endl;
        return false;
    }

    Sim3 refined_sim3;
    if (!estimateSim3FromIndices(
            correspondences,
            best_inliers,
            refined_sim3,
            eff_min_scale,
            eff_max_scale,
            estimate_scale)) {
        std::cout << "LoopClosing: computeSim3 rejected candidate cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << loop_candidate_kf_->id_
                  << " reason=refinement_failed best_inliers=" << best_inliers.size() << std::endl;
        return false;
    }

    best_inliers = computeSim3Inliers(correspondences, refined_sim3, eff_max_residual);
    const double refined_inlier_ratio =
        correspondences.empty() ? 0.0 : static_cast<double>(best_inliers.size()) / correspondences.size();
    const double refined_scale_error = std::abs(refined_sim3.scale() - 1.0);
    const bool metric_scale_ok = loop_closing_internal::isFinalSim3ScaleAcceptable(
        refined_sim3.scale(), has_metric_depth_, metric_scale_tolerance);

    if (best_inliers.size() < static_cast<size_t>(final_min_inliers) ||
        refined_inlier_ratio < final_min_inlier_ratio ||
        !metric_scale_ok) {
        size_t relaxed_inliers = best_inliers.size();
        if (diagnostic_relaxed_residual > eff_max_residual) {
            relaxed_inliers =
                computeSim3Inliers(correspondences, refined_sim3, diagnostic_relaxed_residual).size();
        }
        std::cout << "LoopClosing: computeSim3 rejected candidate cur_kf=" << current_processed_kf_->id_
                  << " cand_kf=" << loop_candidate_kf_->id_
                  << " reason=refined_model_insufficient_support refined_inliers=" << best_inliers.size()
                  << " final_min=" << final_min_inliers
                  << " inlier_ratio=" << refined_inlier_ratio
                  << " final_min_ratio=" << final_min_inlier_ratio
                  << " scale=" << refined_sim3.scale()
                  << " scale_error=" << refined_scale_error
                  << " residual_thresh=" << eff_max_residual
                  << " relaxed_inliers=" << relaxed_inliers
                  << " relaxed_residual_thresh=" << diagnostic_relaxed_residual << std::endl;
        return false;
    }

    corrected_sim3_ = refined_sim3;
    corrected_loop_inlier_count_ = static_cast<int>(best_inliers.size());
    corrected_loop_inlier_ratio_ = refined_inlier_ratio;
    verified_loop_matches_.reserve(best_inliers.size());
    for (const int idx : best_inliers) {
        if (idx < 0 || idx >= static_cast<int>(correspondences.size())) continue;
        verified_loop_matches_.push_back(correspondences[idx].match);
    }

    std::cout << "LoopClosing: Geometric verification succeeded with "
              << verified_loop_matches_.size() << " inliers"
              << " ratio=" << refined_inlier_ratio
              << " | sim3_scale=" << corrected_sim3_.scale() << std::endl;
    return true;
}

void LoopClosing::mergeLandmarks(const Landmark::Ptr& target, const Landmark::Ptr& source) {
    loop_closing_internal::mergeLandmarks(map_, target, source);
}

void LoopClosing::fuseLoopLandmarks() {
    for (const auto& match : verified_loop_matches_) {
        Landmark::Ptr current_lm;
        Landmark::Ptr candidate_lm;
        if (!getMatchedLandmarks(current_processed_kf_, loop_candidate_kf_, match, current_lm, candidate_lm)) {
            continue;
        }

        if (candidate_lm && candidate_lm->isBad()) candidate_lm.reset();
        if (current_lm && current_lm->isBad()) current_lm.reset();

        if (!candidate_lm && !current_lm) continue;

        if (!current_lm && candidate_lm) {
            bool attached = false;
            {
                std::lock_guard<std::mutex> lock(current_processed_kf_->mutex_);
                if (match.queryIdx >= 0 &&
                    match.queryIdx < static_cast<int>(current_processed_kf_->landmarks_.size())) {
                    auto& slot = current_processed_kf_->landmarks_[match.queryIdx];
                    if (!slot || slot->isBad()) {
                        slot = candidate_lm;
                        attached = true;
                    }
                }
            }
            if (attached) {
                candidate_lm->addObservation(current_processed_kf_, static_cast<size_t>(match.queryIdx));
            }
            continue;
        }

        if (current_lm && candidate_lm && current_lm != candidate_lm) {
            mergeLandmarks(current_lm, candidate_lm);
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

    const auto weighting = loop_closing_internal::computeLoopConstraintWeighting(
        corrected_loop_inlier_count_, corrected_loop_inlier_ratio_, has_metric_depth_);

    LoopConstraint constraint;
    constraint.from = current_processed_kf_;
    constraint.to = loop_candidate_kf_;
    constraint.relative_pose = candidate_pose_sim3 * corrected_current_pose_sim3.inverse();
    constraint.translation_weight = weighting.translation_weight;
    constraint.rotation_weight = weighting.rotation_weight;
    constraint.scale_weight = weighting.scale_weight;
    constraint.inlier_count = corrected_loop_inlier_count_;
    constraint.inlier_ratio = corrected_loop_inlier_ratio_;
    loop_constraints_.push_back(constraint);

    std::vector<Optimizer::PoseGraphEdge> optimizer_edges;
    optimizer_edges.reserve(loop_constraints_.size());
    const auto& newest_constraint = loop_constraints_.back();
    for (size_t idx = 0; idx < loop_constraints_.size(); ++idx) {
        const auto& loop_constraint = loop_constraints_[idx];
        double translation_weight = loop_constraint.translation_weight;
        double rotation_weight = loop_constraint.rotation_weight;
        double scale_weight = loop_constraint.scale_weight;

        // When multiple metric-depth closures accumulate, reuse older loop edges
        // only as long as they remain consistent with the current map geometry.
        if (has_metric_depth_ && idx + 1 < loop_constraints_.size()) {
            const auto consistency = evaluateLoopConstraintConsistency(
                loop_constraint.from,
                loop_constraint.to,
                loop_constraint.relative_pose);
            double reuse_decay = computeLoopConstraintReuseDecay(consistency);
            if (consistency.valid && reuse_decay < 0.999) {
                std::cout << "LoopClosing: Reweighting stale loop edge from_kf="
                          << loop_constraint.from->id_
                          << " to_kf=" << loop_constraint.to->id_
                          << " trans_err=" << consistency.translation_error
                          << " scale_err=" << consistency.scale_error
                          << " decay=" << reuse_decay << std::endl;
            }
            const double overlap_decay =
                loop_closing_internal::computeLoopConstraintOverlapDecay(
                    newest_constraint.from->id_,
                    newest_constraint.to->id_,
                    loop_constraint.from->id_,
                    loop_constraint.to->id_,
                    loop_constraint_overlap_window_kf_);
            if (overlap_decay < 0.999) {
                std::cout << "LoopClosing: Downweighting overlapping loop edge from_kf="
                          << loop_constraint.from->id_
                          << " to_kf=" << loop_constraint.to->id_
                          << " newest_from=" << newest_constraint.from->id_
                          << " newest_to=" << newest_constraint.to->id_
                          << " overlap_decay=" << overlap_decay << std::endl;
            }
            reuse_decay = std::min(reuse_decay, overlap_decay);
            translation_weight *= reuse_decay;
            rotation_weight *= reuse_decay;
            scale_weight *= reuse_decay;
        }

        Optimizer::PoseGraphEdge edge;
        edge.from = loop_constraint.from;
        edge.to = loop_constraint.to;
        edge.relative_pose = loop_constraint.relative_pose;
        edge.translation_weight = translation_weight;
        edge.rotation_weight = rotation_weight;
        edge.scale_weight = scale_weight;
        optimizer_edges.push_back(edge);
    }

    std::cout << "LoopClosing: loop edge confidence cur_kf=" << current_processed_kf_->id_
              << " cand_kf=" << loop_candidate_kf_->id_
              << " inliers=" << corrected_loop_inlier_count_
              << " ratio=" << corrected_loop_inlier_ratio_
              << " weight=" << weighting.translation_weight << std::endl;

    // Block tracking before modifying poses/landmarks
    map_->loop_correcting_.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    Optimizer::poseGraphOptimization(map_, optimizer_edges, 90, has_metric_depth_);

    fuseLoopLandmarks();

    const auto& all_keyframes = map_->getAllKeyframes();
    for (const auto& kv : all_keyframes) {
        if (kv.second) kv.second->updateConnections();
    }

    map_->loop_correcting_.store(false);

    // Notify tracking to recompute its current frame pose against the corrected map
    if (on_loop_corrected_) {
        on_loop_corrected_();
    }

    std::cout << "LoopClosing: Applied global pose graph with "
              << optimizer_edges.size() << " loop constraints and global BA." << std::endl;
}

}
