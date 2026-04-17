#include "backend/local_mapping.h"
#include "backend/optimizer.h"
#include "loop_closing/loop_closing.h"
#include "core/landmark.h"
#include "core/keyframe.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

namespace svslam {

LocalMapping::LocalMapping(Map::Ptr map) : map_(map) {}

void LocalMapping::setLoopClosing(std::shared_ptr<LoopClosing> loop_closing) {
    loop_closing_ = loop_closing;
}

void LocalMapping::insertKeyframe(Keyframe::Ptr kf) {
    std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
    new_keyframes_.push_back(kf);
    cv_new_keyframes_.notify_one();
}

void LocalMapping::requestStop() {
    stop_requested_ = true;
    cv_new_keyframes_.notify_one();
}

void LocalMapping::run() {
    std::cout << "LocalMapping thread started." << std::endl;
    while (!stop_requested_) {
        // Wait for new keyframes
        {
            std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
            if (new_keyframes_.empty()) {
                cv_new_keyframes_.wait(lock);
            }
            if (stop_requested_) break;
        }
        
        processPendingWork();
    }
    std::cout << "LocalMapping thread stopped." << std::endl;
}

void LocalMapping::processPendingWork() {
    while (checkNewKeyframes()) {
        processNewKeyframe();
        mapPointCulling();
        createNewMapPoints();
        current_processed_kf_->updateConnections();
        // Skip BA while loop closing is correcting poses/landmarks
        if (!map_->loop_correcting_.load()) {
            optimization();
        }
    }
}

bool LocalMapping::checkNewKeyframes() {
    std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
    return !new_keyframes_.empty();
}

void LocalMapping::processNewKeyframe() {
    {
        std::unique_lock<std::mutex> lock(mutex_new_keyframes_);
        current_processed_kf_ = new_keyframes_.front();
        new_keyframes_.pop_front();
    }
    
    // Update connections (Covisibility Graph)
    current_processed_kf_->updateConnections();
    
    // Add to Map if not already added
    map_->addKeyframe(current_processed_kf_);
    ++processed_keyframe_count_;
    
    std::cout << "LocalMapping: Processed Keyframe " << current_processed_kf_->id_ 
              << ". Connected KFs: " << current_processed_kf_->connected_keyframes_.size() << std::endl;

    // Pass to Loop Closing
    if (loop_closing_) {
        loop_closing_->insertKeyframe(current_processed_kf_);
    }
}

void LocalMapping::removeLandmark(const Landmark::Ptr& lm) {
    if (!lm || lm->isBad()) return;

    std::vector<std::pair<Keyframe::Ptr, size_t>> observations;
    {
        std::unique_lock<std::mutex> lock(lm->mutex_);
        for (const auto& obs : lm->observations_) {
            auto kf = obs.first.lock();
            if (kf) {
                observations.push_back({kf, obs.second});
            }
        }
        lm->observations_.clear();
        lm->setBad();
    }

    for (const auto& obs : observations) {
        auto kf = obs.first;
        const size_t idx = obs.second;
        std::unique_lock<std::mutex> lock(kf->mutex_);
        if (idx < kf->landmarks_.size() && kf->landmarks_[idx] == lm) {
            kf->landmarks_[idx] = nullptr;
        }
    }

    map_->removeLandmark(lm);
}

void LocalMapping::mapPointCulling() {
    int culled = 0;
    const bool mono_mode = current_processed_kf_ && current_processed_kf_->depth_image_.empty();
    for (auto it = recent_landmarks_.begin(); it != recent_landmarks_.end();) {
        const Landmark::Ptr lm = it->first;
        const unsigned long created_at = it->second;

        if (!lm || lm->isBad()) {
            it = recent_landmarks_.erase(it);
            continue;
        }

        size_t observation_count = 0;
        {
            std::unique_lock<std::mutex> lock(lm->mutex_);
            observation_count = lm->observations_.size();
        }

        const unsigned long age = processed_keyframe_count_ > created_at
            ? processed_keyframe_count_ - created_at
            : 0;
        const bool should_cull = mono_mode
            ? ((observation_count <= 1 && age >= 1) ||
               (observation_count < 3 && age >= 1))
            : ((observation_count <= 1 && age >= 1) ||
               (observation_count < 3 && age >= 2));
        const bool no_longer_recent = (observation_count >= 3) || (age >= 3);

        if (should_cull) {
            removeLandmark(lm);
            ++culled;
            it = recent_landmarks_.erase(it);
            continue;
        }

        if (no_longer_recent) {
            it = recent_landmarks_.erase(it);
            continue;
        }

        ++it;
    }
    
    if (culled > 0)
        std::cout << "LocalMapping: Culled " << culled << " map points." << std::endl;
}

void LocalMapping::createNewMapPoints() {
    // First: create landmarks directly from depth for unmatched keypoints
    if (!current_processed_kf_->depth_image_.empty()) {
        SE3 T_wc = current_processed_kf_->T_cw_.inverse();
        int depth_created = 0;
        static unsigned long depth_mapping_lm_id = 400000;

        for (size_t i = 0; i < current_processed_kf_->keypoints_.size(); ++i) {
            if (current_processed_kf_->landmarks_[i]) continue;

            const auto& kp = current_processed_kf_->keypoints_[i];
            float depth = current_processed_kf_->getDepth(kp.pt.x, kp.pt.y);
            if (depth <= 0.0f || depth > 10.0f) continue;

            Vec3 p_norm = current_processed_kf_->camera_->unproject(Vec2(kp.pt.x, kp.pt.y));
            Vec3 p_cam = p_norm * static_cast<double>(depth);
            Vec3 p_w = T_wc * p_cam;

            auto lm = std::make_shared<Landmark>(depth_mapping_lm_id++, p_w);
            lm->addObservation(current_processed_kf_, i);
            lm->descriptor_ = current_processed_kf_->descriptors_.row(i).clone();

            current_processed_kf_->landmarks_[i] = lm;
            map_->addLandmark(lm);
            recent_landmarks_.push_back({lm, processed_keyframe_count_});
            depth_created++;
        }

        if (depth_created > 0) {
            std::cout << "LocalMapping: Created " << depth_created << " landmarks from depth" << std::endl;
        }
    }

    // Then: triangulate new points between current_processed_kf_ and its neighbors
    int nn = 15;
    std::vector<Keyframe::Ptr> neighbors = current_processed_kf_->getBestCovisibilityKeyframes(nn);

    int valid_landmarks = 0;
    for (const auto& lm : current_processed_kf_->landmarks_) {
        if (lm && !lm->isBad()) {
            ++valid_landmarks;
        }
    }

    // Fallback: if covisibility graph has no neighbors, use all keyframes in the map
    if (neighbors.empty()) {
        const bool sparse_mono_without_support =
            current_processed_kf_->depth_image_.empty() &&
            valid_landmarks < 20;
        if (sparse_mono_without_support) {
            std::cout << "LocalMapping::createNewMapPoints: skipping all-KF fallback for sparse mono KF "
                      << current_processed_kf_->id_
                      << " (valid_landmarks=" << valid_landmarks << ")" << std::endl;
            return;
        }

        const auto& all_kfs = map_->getAllKeyframes();
        for (const auto& kv : all_kfs) {
            if (kv.second && kv.second != current_processed_kf_) {
                neighbors.push_back(kv.second);
            }
        }
        std::cout << "LocalMapping::createNewMapPoints: covisibility empty, using all "
                  << neighbors.size() << " KFs as neighbors" << std::endl;
    }

    if (neighbors.empty()) return;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    SE3 T_cw1 = current_processed_kf_->T_cw_;

    // Get unmatched keypoints in current KF
    std::vector<int> unmatched_indices_1;
    for (size_t i=0; i < current_processed_kf_->keypoints_.size(); ++i) {
        if (!current_processed_kf_->landmarks_[i]) {
            unmatched_indices_1.push_back(i);
        }
    }

    if (unmatched_indices_1.empty()) return;

    // Convert descriptors to Mat for query
    cv::Mat descriptors_1;
    for (int idx : unmatched_indices_1) {
        descriptors_1.push_back(current_processed_kf_->descriptors_.row(idx));
    }

    Eigen::Vector3d Ow1 = T_cw1.so3().inverse() * -T_cw1.translation();

    int total_new_points = 0;

    for (auto& neighbor : neighbors) {
        if (!neighbor) continue;
        SE3 T_cw2 = neighbor->T_cw_;
        Eigen::Vector3d Ow2 = T_cw2.so3().inverse() * -T_cw2.translation();

        const bool mono_pair =
            current_processed_kf_->depth_image_.empty() &&
            neighbor->depth_image_.empty();

        // Monocular triangulation is much less stable on very short baselines.
        const double min_baseline = mono_pair ? 0.02 : 0.01;
        double baseline = (Ow1 - Ow2).norm();
        if (baseline < min_baseline) continue;

        std::vector<int> unmatched_indices_2;
        cv::Mat descriptors_2;
        for (size_t i=0; i < neighbor->keypoints_.size(); ++i) {
            if (!neighbor->landmarks_[i]) {
                unmatched_indices_2.push_back(i);
                descriptors_2.push_back(neighbor->descriptors_.row(i));
            }
        }

        if (descriptors_2.empty()) continue;

        // Use knnMatch + ratio test for better matching quality
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

        std::vector<cv::DMatch> matches;
        for (const auto& ms : knn_matches) {
            if (ms.size() < 2) continue;
            if (ms[0].distance > 64.0f) continue;
            if (ms[0].distance >= 0.8f * ms[1].distance) continue;
            matches.push_back(ms[0]);
        }

        int new_points_this_neighbor = 0;
        int dbg_already_matched = 0;
        int dbg_w_invalid = 0;
        int dbg_depth_fail = 0;
        int dbg_bounds_fail = 0;
        int dbg_reproj_fail = 0;
        int dbg_nonfinite = 0;
        for (auto& m : matches) {
            
            int idx1 = unmatched_indices_1[m.queryIdx];
            int idx2 = unmatched_indices_2[m.trainIdx];

            if (current_processed_kf_->landmarks_[idx1] || neighbor->landmarks_[idx2]) {
                dbg_already_matched++;
                continue;
            }

            // Triangulate
            std::vector<cv::Point2f> pts1, pts2;
            pts1.push_back(current_processed_kf_->keypoints_[idx1].pt);
            pts2.push_back(neighbor->keypoints_[idx2].pt);

            cv::Mat pt_4d;

            Eigen::Matrix<double, 3, 4> mat1 = T_cw1.matrix3x4();
            Eigen::Matrix<double, 3, 4> mat2 = T_cw2.matrix3x4();
            cv::Mat T1_cv, T2_cv;
            cv::eigen2cv(mat1, T1_cv);
            cv::eigen2cv(mat2, T2_cv);

            cv::Mat P1 = current_processed_kf_->camera_->K() * T1_cv;
            cv::Mat P2 = neighbor->camera_->K() * T2_cv;

            cv::triangulatePoints(P1, P2, pts1, pts2, pt_4d);

            // triangulatePoints returns CV_32F regardless of input types
            float w = pt_4d.at<float>(3, 0);
            if (w == 0.0f) { dbg_w_invalid++; continue; }

            cv::Point3f pt_w(pt_4d.at<float>(0, 0) / w,
                             pt_4d.at<float>(1, 0) / w,
                             pt_4d.at<float>(2, 0) / w);

            Vec3 P(pt_w.x, pt_w.y, pt_w.z);

            if (!std::isfinite(P.x()) || !std::isfinite(P.y()) || !std::isfinite(P.z())) {
                dbg_nonfinite++;
                continue;
            }

            const double max_pos = 50.0;
            if (std::abs(P.x()) > max_pos || std::abs(P.y()) > max_pos || std::abs(P.z()) > max_pos) {
                dbg_bounds_fail++;
                continue;
            }

            double d1 = (T_cw1 * P)[2];
            double d2 = (T_cw2 * P)[2];

            // Check positive depth in both views and reasonable depth range
            const double min_depth = 0.1;
            const double max_depth = 20.0;  // Reasonable for indoor scenes
            if (!(d1 > min_depth && d1 < max_depth && d2 > min_depth && d2 < max_depth)) {
                dbg_depth_fail++;
                continue;
            }
            {
                 // Reprojection error check
                 Vec2 proj1 = current_processed_kf_->camera_->project(T_cw1 * P);
                 Vec2 proj2 = neighbor->camera_->project(T_cw2 * P);
                 const auto& kp1 = current_processed_kf_->keypoints_[idx1].pt;
                 const auto& kp2 = neighbor->keypoints_[idx2].pt;
                 double err1 = std::sqrt((kp1.x - proj1[0]) * (kp1.x - proj1[0]) + (kp1.y - proj1[1]) * (kp1.y - proj1[1]));
                 double err2 = std::sqrt((kp2.x - proj2[0]) * (kp2.x - proj2[0]) + (kp2.y - proj2[1]) * (kp2.y - proj2[1]));
                 if (err1 > 8.0 || err2 > 8.0) { dbg_reproj_fail++; continue; }

                 // Success - create map point
                 static unsigned long lm_id_counter = 10000; // TODO: Global ID
                 auto lm = std::make_shared<Landmark>(lm_id_counter++, P);

                 lm->addObservation(current_processed_kf_, idx1);
                 lm->addObservation(neighbor, idx2);
                 lm->descriptor_ = current_processed_kf_->descriptors_.row(idx1).clone();

                 // Add to keyframes. Take each kf->mutex_ separately so we do
                 // not hold two keyframe locks at once (avoids lock-order
                 // inversion vs other paths that take these locks singly).
                 {
                     std::lock_guard<std::mutex> lock(current_processed_kf_->mutex_);
                     current_processed_kf_->landmarks_[idx1] = lm;
                 }
                 {
                     std::lock_guard<std::mutex> lock(neighbor->mutex_);
                     neighbor->landmarks_[idx2] = lm;
                 }

                 map_->addLandmark(lm);
                 recent_landmarks_.push_back({lm, processed_keyframe_count_});
                 new_points_this_neighbor++;
                 total_new_points++;
            }
        }

        if (matches.size() > 0) {
            std::cout << "LocalMapping: KF " << neighbor->id_
                      << " baseline=" << baseline
                      << " matches=" << matches.size()
                      << " new=" << new_points_this_neighbor
                      << " (already=" << dbg_already_matched
                      << " w0=" << dbg_w_invalid
                      << " nonfinite=" << dbg_nonfinite
                      << " bounds=" << dbg_bounds_fail
                      << " depth=" << dbg_depth_fail
                      << " reproj=" << dbg_reproj_fail
                      << ")" << std::endl;
        }
    }

    std::cout << "LocalMapping::createNewMapPoints: Total new points = " << total_new_points
              << " (map now has " << map_->getAllLandmarks().size() << " landmarks)" << std::endl;
}

void LocalMapping::optimization() {
    // Local Bundle Adjustment
    // 1. Setup local keyframes: current KF and its neighbors
    std::vector<Keyframe::Ptr> local_keyframes;
    local_keyframes.push_back(current_processed_kf_);

    auto neighbors = current_processed_kf_->getBestCovisibilityKeyframes(20);
    for (auto& kf : neighbors) {
        if (!kf) continue;
        local_keyframes.push_back(kf);
    }

    // 2. Setup local map points: all points observed by local keyframes
    // NOTE: Avoid using std::set<shared_ptr<...>> here because it orders by pointer value,
    // which varies run-to-run and can introduce non-determinism in BA.
    std::vector<Landmark::Ptr> local_landmarks;
    for (auto& kf : local_keyframes) {
        for (auto& lm : kf->landmarks_) {
            if (lm && !lm->isBad()) {
                local_landmarks.push_back(lm);
            }
        }
    }
    std::sort(local_landmarks.begin(), local_landmarks.end(),
              [](const Landmark::Ptr& a, const Landmark::Ptr& b) {
                  return a->id_ < b->id_;
              });
    local_landmarks.erase(std::unique(local_landmarks.begin(), local_landmarks.end(),
                                      [](const Landmark::Ptr& a, const Landmark::Ptr& b) {
                                          return a->id_ == b->id_;
                                      }),
                          local_landmarks.end());

    // Limit BA size to keep it fast
    const size_t max_ba_landmarks = 800;
    if (local_landmarks.size() > max_ba_landmarks) {
        // Keep landmarks with most observations (most constrained)
        std::sort(local_landmarks.begin(), local_landmarks.end(),
            [](const Landmark::Ptr& a, const Landmark::Ptr& b) {
                const size_t a_obs = a ? a->observations_.size() : 0;
                const size_t b_obs = b ? b->observations_.size() : 0;
                if (a_obs != b_obs) return a_obs > b_obs;
                const unsigned long a_id = a ? a->id_ : 0;
                const unsigned long b_id = b ? b->id_ : 0;
                return a_id < b_id;
            });
        local_landmarks.resize(max_ba_landmarks);
    }

    std::cout << "LocalMapping: BA on " << local_keyframes.size() << " KFs and " << local_landmarks.size() << " LMs." << std::endl;

    if (local_keyframes.size() < 2 || local_landmarks.size() < 10) return;

    Optimizer::bundleAdjustment(local_keyframes, local_landmarks, 20);

    // Notify Tracking that BA is complete so it can recompute current frame pose
    if (on_ba_completed_) {
        on_ba_completed_();
    }
}

}
