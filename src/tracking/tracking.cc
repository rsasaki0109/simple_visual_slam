#include "tracking/tracking.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include "core/keyframe.h"
#include "core/landmark.h"

namespace svslam {

Tracking::Tracking() : state_(TrackingState::NO_IMAGES_YET) {
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void Tracking::setMap(std::shared_ptr<Map> map) {
    map_ = map;
}

void Tracking::setLocalMapping(std::shared_ptr<LocalMapping> local_mapping) {
    local_mapping_ = local_mapping;
}

bool Tracking::addFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    if (state_ == TrackingState::NO_IMAGES_YET) {
        state_ = TrackingState::NOT_INITIALIZED;
    }

    bool success = false;
    if (state_ == TrackingState::NOT_INITIALIZED) {
        success = initialize();
    } else {
        success = track();
    }

    last_frame_ = current_frame_;
    return success;
}

bool Tracking::initialize() {
    if (!initial_frame_) {
        // First frame
        initial_frame_ = current_frame_;
        // Set identity pose
        initial_frame_->setPose(SE3());
        
        // Initialize initializer
        initializer_ = std::make_shared<Initializer>(initial_frame_);
        
        std::cout << "Tracking: Initial Frame set (ID: " << initial_frame_->id_ << ")" << std::endl;
        return true; 
    } else {
        // Second frame, try to initialize
        if (initializer_->initialize(current_frame_)) {
            std::cout << "Tracking: Initialization SUCCESS!" << std::endl;
            
            // 1. Create Keyframes
            auto kf_init = std::make_shared<Keyframe>(initial_frame_);
            auto kf_cur = std::make_shared<Keyframe>(current_frame_);
            
            // Set Pose for current (T_cw)
            // Initializer returns T_c1_c2 which we defined as T_c2_c1 (Pose of 2 w.r.t 1)
            // T_cw_cur = T_c2_c1 * T_cw_ref (where T_cw_ref is Identity)
            current_frame_->setPose(initializer_->T_c1_c2_);
            kf_cur->T_cw_ = current_frame_->getPose();
            
            std::cout << "Tracking: Initialized Pose T_c2_c1: \n" << current_frame_->getPose().matrix() << std::endl;
            
            // 2. Create MapPoints
            size_t tri_true = 0;
            size_t inserted = 0;
            size_t rejected_nonfinite = 0;
            size_t rejected_nonpositive_z = 0;
            size_t rejected_absmax = 0;

            double z_min = std::numeric_limits<double>::infinity();
            double z_max = -std::numeric_limits<double>::infinity();
            double norm_max = 0.0;

            for (size_t i = 0; i < initializer_->is_triangulated_.size(); ++i) {
                if (initializer_->is_triangulated_[i]) {
                    tri_true++;
                    // Create Landmark
                    cv::Point3f pt3d = initializer_->triangulated_points_[i]; // In Ref frame
                    if (!std::isfinite(pt3d.x) || !std::isfinite(pt3d.y) || !std::isfinite(pt3d.z)) {
                        rejected_nonfinite++;
                        continue;
                    }
                    if (pt3d.z <= 0.0f) {
                        rejected_nonpositive_z++;
                        continue;
                    }
                    const float abs_max = 1e4f;
                    if (std::abs(pt3d.x) > abs_max || std::abs(pt3d.y) > abs_max || std::abs(pt3d.z) > abs_max) {
                        rejected_absmax++;
                        continue;
                    }

                    z_min = std::min(z_min, static_cast<double>(pt3d.z));
                    z_max = std::max(z_max, static_cast<double>(pt3d.z));
                    const double nrm = std::sqrt(static_cast<double>(pt3d.x) * pt3d.x + static_cast<double>(pt3d.y) * pt3d.y + static_cast<double>(pt3d.z) * pt3d.z);
                    norm_max = std::max(norm_max, nrm);

                    Vec3 pos_w(pt3d.x, pt3d.y, pt3d.z); // Ref is World
                    
                    auto lm = std::make_shared<Landmark>(i, pos_w); // ID? Use global ID counter later
                    
                    // Add observations
                    // We need to know feature index in keyframe
                    int idx_ref = initializer_->matches_[i].queryIdx;
                    int idx_cur = initializer_->matches_[i].trainIdx;
                    
                    lm->addObservation(kf_init, idx_ref);
                    lm->addObservation(kf_cur, idx_cur);
                    
                    // Set descriptor (using reference frame)
                    lm->descriptor_ = initial_frame_->descriptors_.row(idx_ref).clone();

                    // Add landmarks to keyframes
                    kf_init->landmarks_[idx_ref] = lm;
                    kf_cur->landmarks_[idx_cur] = lm;
                    
                    // Update Frames as well so they are tracked
                    initial_frame_->landmarks_[idx_ref] = lm;
                    current_frame_->landmarks_[idx_cur] = lm;
                    
                    // Add to map
                    if (map_) {
                        map_->addLandmark(lm);
                        inserted++;
                    }
                }
            }

            std::cout << "Tracking: Init triangulation tri_true=" << tri_true
                      << " inserted=" << inserted
                      << " rejected_nonfinite=" << rejected_nonfinite
                      << " rejected_nonpositive_z=" << rejected_nonpositive_z
                      << " rejected_absmax=" << rejected_absmax
                      << " z_min=" << z_min
                      << " z_max=" << z_max
                      << " norm_max=" << norm_max
                      << std::endl;
            
            if (map_) {
                map_->addKeyframe(kf_init);
                map_->addKeyframe(kf_cur);

                std::cout << "Tracking: Map after init: keyframes=" << map_->getAllKeyframes().size()
                          << " landmarks=" << map_->getAllLandmarks().size() << std::endl;
            }
            
            state_ = TrackingState::OK;
            return true;
        } else {
            std::cout << "Tracking: Initialization failed. Retrying..." << std::endl;
            // Reset? Or just keep trying with new current vs old initial?
            // ORB-SLAM replaces initial if not enough disparity
            return false;
        }
    }
}

bool Tracking::track() {
    // 1. Motion Model Prediction
    if (last_frame_) {
        // T_current = velocity * T_last
        SE3 T_cw_pred = velocity_ * last_frame_->getPose();
        current_frame_->setPose(T_cw_pred);
    }

    // 2. Track Reference Keyframe (Frame-to-Frame matching for now)
    bool ref_tracking_ok = trackReferenceKeyframe();
    
    if (!ref_tracking_ok) {
        std::cout << "Reference tracking failed!" << std::endl;
        return false;
    }
    
    // 3. Track Local Map
    bool local_map_ok = trackLocalMap();
    if (!local_map_ok) {
        std::cout << "Local Map tracking failed!" << std::endl;
        // Proceed with what we have? Or declare lost?
        // For now, if ref tracking was ok, we might be fine, but let's be strict.
        // return false; 
    }

    // Update velocity for next frame (Constant velocity model)
    if (last_frame_) {
        // velocity = T_current * T_last^-1
        velocity_ = current_frame_->getPose() * last_frame_->getPose().inverse();
    }
    
    // 4. Check if we need a new Keyframe
    if (needNewKeyframe()) {
        std::cout << "Tracking: Insert New Keyframe" << std::endl;
        // Create new Keyframe
        auto kf = std::make_shared<Keyframe>(current_frame_);
        
        if (local_mapping_) {
            local_mapping_->insertKeyframe(kf);
        } else {
            map_->addKeyframe(kf);
        }
        
        // In a real system, we would trigger LocalMapping here
        // For now, just add to map and maybe triangulate some points?
        // Since we don't have LocalMapping thread yet, this is minimal.
    }
    
    return true;
}

bool Tracking::needNewKeyframe() {
    if (!map_) return false;
    
    // Simple heuristic:
    // 1. Enough tracked points? (If too few, we might be lost, but let's say if < 20 we need info)
    // 2. Or if too many frames passed since last KF?
    
    // Count tracked landmarks
    int n_tracked = 0;
    // We don't have a direct list of tracked landmarks in current_frame_ easily accessible 
    // unless we check which ones projected successfully in trackLocalMap.
    // But trackLocalMap didn't strictly associate landmarks back to current_frame_ features 
    // except by updating pose. 
    // Ideally trackLocalMap should update current_frame_->landmarks_ with matched map points.
    
    // Let's assume for now we don't insert KFs unless we implement proper association update.
    // Wait, trackLocalMap used solvePnPRansac but didn't store the matches in frame.
    // We should fix trackLocalMap to store matches.
    
    return false; // Stub for now until trackLocalMap is improved
}

bool Tracking::trackLocalMap() {
    if (!map_) return false;

    // 1. Get all landmarks (Naive local map: all map points)
    // In real SLAM, we should only select those in view frustum and connected to local keyframes.
    auto landmarks = map_->getAllLandmarks();
    
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<std::shared_ptr<Landmark>> matched_landmarks; // Keep track of LM for each point
    std::vector<int> matched_kp_indices; // Keep track of KP index for each point

    std::vector<bool> keypoint_already_matched(current_frame_->keypoints_.size(), false);

    bool used_global_fallback = false;

    // For fallback matching, only consider landmarks that are in the current view frustum
    cv::Mat visible_lm_descs;
    std::vector<Landmark::Ptr> visible_lm_list;
    std::vector<cv::Point3f> visible_lm_pts;
    
    std::cout << "TrackLocalMap: Landmarks to check: " << landmarks.size() << std::endl;

    // 2. Project and Match
    int matches_found = 0;
    int visible_points = 0;
    int skipped_nonfinite = 0;
    int skipped_behind_or_close = 0;
    int skipped_oob = 0;
    for (auto& pair : landmarks) {
        auto lm = pair.second;
        if (!lm) continue;
        if (lm->descriptor_.empty()) continue;
        
        // Project
        Vec3 pos_w = lm->getPos();
        if (!std::isfinite(pos_w[0]) || !std::isfinite(pos_w[1]) || !std::isfinite(pos_w[2])) {
            skipped_nonfinite++;
            continue;
        }
        Vec3 pos_c = current_frame_->getPose() * pos_w; // T_cw * pos_w
        
        if (pos_c[2] <= 0.1) {
            skipped_behind_or_close++;
            continue; // Behind camera or too close
        }
        
        Vec2 px = current_frame_->camera_->project(pos_c);
        
        // Check bounds
        if (px[0] < 0 || px[0] >= current_frame_->image_.cols ||
            px[1] < 0 || px[1] >= current_frame_->image_.rows) {
            skipped_oob++;
            continue;
        }
            
        visible_points++;

        // Cache visible landmarks for fallback matching
        visible_lm_descs.push_back(lm->descriptor_);
        visible_lm_list.push_back(lm);
        visible_lm_pts.push_back(cv::Point3f(pos_w[0], pos_w[1], pos_w[2]));
        
        // Search for match in current frame features
        // Simple search: look for features near px
        int best_idx = -1;
        double best_dist = 80.0; // Descriptor distance threshold (Hamming)
        const double search_radius_sq = 200.0 * 200.0;
        
        // Radius search (naive O(N) per landmark)
        // Ideally should use grid search
        for (size_t i = 0; i < current_frame_->keypoints_.size(); ++i) {
             if (keypoint_already_matched[i]) continue;
             const auto& kp = current_frame_->keypoints_[i];
             double dist_spatial = (kp.pt.x - px[0])*(kp.pt.x - px[0]) + (kp.pt.y - px[1])*(kp.pt.y - px[1]);
             
             if (dist_spatial < search_radius_sq) {
                 // Check descriptor distance
                 double dist_desc = cv::norm(current_frame_->descriptors_.row(i), lm->descriptor_, cv::NORM_HAMMING);
                 if (dist_desc < best_dist) {
                     best_dist = dist_desc;
                     best_idx = i;
                 }
             }
        }
        
        if (best_idx != -1) {
            object_points.push_back(cv::Point3f(pos_w[0], pos_w[1], pos_w[2]));
            image_points.push_back(current_frame_->keypoints_[best_idx].pt);
            
            matched_landmarks.push_back(lm);
            matched_kp_indices.push_back(best_idx);
            keypoint_already_matched[best_idx] = true;
            
            matches_found++;
        }
    }
    
    std::cout << "TrackLocalMap: Visible: " << visible_points
              << ", Matches: " << matches_found
              << " (skipped nonfinite=" << skipped_nonfinite
              << " behind/close=" << skipped_behind_or_close
              << " oob=" << skipped_oob
              << ")" << std::endl;

    // Fallback: if pose is too noisy, projection-based matching may find 0.
    // In that case, do global descriptor matching (landmark descriptor -> current descriptors)
    // to bootstrap PnP.
    if (object_points.size() < 10) {
        if (!visible_lm_descs.empty() && !current_frame_->descriptors_.empty()) {
            cv::BFMatcher bf(cv::NORM_HAMMING);
            std::vector<std::vector<cv::DMatch>> knn;
            bf.knnMatch(visible_lm_descs, current_frame_->descriptors_, knn, 2);

            std::vector<bool> kp_used(current_frame_->keypoints_.size(), false);
            std::vector<bool> lm_used(visible_lm_list.size(), false);

            struct MatchCandidate {
                int lm_idx;
                int kp_idx;
                float dist;
            };
            std::vector<MatchCandidate> candidates;

            for (const auto& ms : knn) {
                if (ms.size() < 2) continue;
                const auto& m1 = ms[0];
                const auto& m2 = ms[1];

                // Lowe ratio test + absolute distance gate
                if (m1.distance > 70.0f) continue;
                if (m1.distance >= 0.75f * m2.distance) continue;

                if (m1.queryIdx < 0 || m1.queryIdx >= static_cast<int>(visible_lm_list.size())) continue;
                if (m1.trainIdx < 0 || m1.trainIdx >= static_cast<int>(current_frame_->keypoints_.size())) continue;
                if (lm_used[m1.queryIdx] || kp_used[m1.trainIdx]) continue;

                candidates.push_back({m1.queryIdx, m1.trainIdx, m1.distance});
            }

            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.dist < b.dist;
            });

            const size_t max_keep = 200;
            for (size_t i = 0; i < candidates.size() && i < max_keep; ++i) {
                const auto& c = candidates[i];
                if (lm_used[c.lm_idx] || kp_used[c.kp_idx]) continue;
                object_points.push_back(visible_lm_pts[c.lm_idx]);
                image_points.push_back(current_frame_->keypoints_[c.kp_idx].pt);
                matched_landmarks.push_back(visible_lm_list[c.lm_idx]);
                matched_kp_indices.push_back(c.kp_idx);
                lm_used[c.lm_idx] = true;
                kp_used[c.kp_idx] = true;
            }

            // Geometric gating using current pose: keep only matches consistent with projection.
            // This helps improve inlier ratio before PnP.
            if (!object_points.empty()) {
                std::vector<cv::Point3f> obj_f;
                std::vector<cv::Point2f> img_f;
                std::vector<Landmark::Ptr> lm_f;
                std::vector<int> kp_f;

                const double gate_sq = 80.0 * 80.0;
                const SE3 T_cw_est = current_frame_->getPose();

                for (size_t i = 0; i < object_points.size(); ++i) {
                    const auto& Pw = object_points[i];
                    Vec3 p_w(Pw.x, Pw.y, Pw.z);
                    Vec3 p_c = T_cw_est * p_w;
                    if (p_c[2] <= 0.1) continue;
                    Vec2 proj = current_frame_->camera_->project(p_c);
                    const auto& uv = image_points[i];
                    const double dx = uv.x - proj[0];
                    const double dy = uv.y - proj[1];
                    const double e2 = dx * dx + dy * dy;
                    if (e2 > gate_sq) continue;

                    obj_f.push_back(Pw);
                    img_f.push_back(uv);
                    lm_f.push_back(matched_landmarks[i]);
                    kp_f.push_back(matched_kp_indices[i]);
                }

                object_points.swap(obj_f);
                image_points.swap(img_f);
                matched_landmarks.swap(lm_f);
                matched_kp_indices.swap(kp_f);
            }

            used_global_fallback = true;
            std::cout << "TrackLocalMap: Fallback global matches: " << object_points.size() << std::endl;
        }
    }
    
    if (object_points.size() < 10) return false;

    // Diagnostics: check correspondence sanity before PnP.
    // This helps identify issues like broken 3D points, bad scale, or inconsistent 3D-2D pairs.
    {
        size_t n = object_points.size();
        size_t nan_or_inf = 0;
        size_t non_positive_depth = 0;

        double z_min = std::numeric_limits<double>::infinity();
        double z_max = -std::numeric_limits<double>::infinity();
        double z_sum = 0.0;
        size_t z_cnt = 0;

        double err_min = std::numeric_limits<double>::infinity();
        double err_max = -std::numeric_limits<double>::infinity();
        double err_sum = 0.0;
        size_t err_cnt = 0;

        const SE3 T_cw_est = current_frame_->getPose();

        for (size_t i = 0; i < n; ++i) {
            const auto& Pw = object_points[i];
            if (!std::isfinite(Pw.x) || !std::isfinite(Pw.y) || !std::isfinite(Pw.z)) {
                nan_or_inf++;
                continue;
            }

            Vec3 p_w(Pw.x, Pw.y, Pw.z);
            Vec3 p_c = T_cw_est * p_w;

            const double z = p_c[2];
            if (!std::isfinite(z)) {
                nan_or_inf++;
                continue;
            }
            if (z <= 0.0) {
                non_positive_depth++;
                continue;
            }

            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
            z_sum += z;
            z_cnt++;

            Vec2 proj = current_frame_->camera_->project(p_c);
            const auto& uv = image_points[i];
            const double dx = uv.x - proj[0];
            const double dy = uv.y - proj[1];
            const double e = std::sqrt(dx * dx + dy * dy);
            if (std::isfinite(e)) {
                err_min = std::min(err_min, e);
                err_max = std::max(err_max, e);
                err_sum += e;
                err_cnt++;
            }
        }

        const double z_mean = (z_cnt > 0) ? (z_sum / static_cast<double>(z_cnt)) : std::numeric_limits<double>::quiet_NaN();
        const double err_mean = (err_cnt > 0) ? (err_sum / static_cast<double>(err_cnt)) : std::numeric_limits<double>::quiet_NaN();

        std::cout << "TrackLocalMap: CorrStats n=" << n
                  << " nan_inf=" << nan_or_inf
                  << " nonpos_z=" << non_positive_depth
                  << " z[min/mean/max]=" << z_min << "/" << z_mean << "/" << z_max
                  << " reproj_err_px[min/mean/max]=" << err_min << "/" << err_mean << "/" << err_max
                  << " used_global_fallback=" << (used_global_fallback ? 1 : 0)
                  << std::endl;
    }
    
    // 3. Optimize Pose (solvePnPRansac)
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    
    // Initial guess from motion model
    Eigen::Vector3d t = current_frame_->getPose().translation();
    Eigen::Matrix3d R = current_frame_->getPose().rotationMatrix();
    cv::Mat R_cv, t_cv;
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    cv::eigen2cv(t, tvec);
    
    // If correspondences come from global descriptor matching, do not trust the motion-model pose as an initial guess.
    const bool use_extrinsic_guess = !used_global_fallback;

    enum class PnpMethod { EPNP, P3P, ITERATIVE };
    auto try_pnp = [&](PnpMethod method) -> bool {
        int flag = cv::SOLVEPNP_EPNP;
        const char* name = "EPNP";
        if (method == PnpMethod::P3P) {
            flag = cv::SOLVEPNP_P3P;
            name = "P3P";
        } else if (method == PnpMethod::ITERATIVE) {
            flag = cv::SOLVEPNP_ITERATIVE;
            name = "ITERATIVE";
        }

        std::vector<int> tmp_inliers;
        cv::Mat rvec_tmp = rvec.clone();
        cv::Mat tvec_tmp = tvec.clone();

        bool ok = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                     rvec_tmp, tvec_tmp, use_extrinsic_guess,
                                     200, 20.0, 0.99, tmp_inliers, flag);
        if (ok) {
            rvec = rvec_tmp;
            tvec = tvec_tmp;
            inliers.swap(tmp_inliers);
            std::cout << "TrackLocalMap: PnP method=" << name << " inliers=" << inliers.size() << std::endl;
            return true;
        }
        return false;
    };

    bool success = try_pnp(PnpMethod::EPNP) || try_pnp(PnpMethod::P3P) || try_pnp(PnpMethod::ITERATIVE);
                                      
    if (success) {
        std::cout << "TrackLocalMap: PnP Success, inliers: " << inliers.size() << std::endl;
        // Update pose
        cv::Mat R_new;
        cv::Rodrigues(rvec, R_new);
        Eigen::Matrix3d R_eig;
        Eigen::Vector3d t_eig;
        cv::cv2eigen(R_new, R_eig);
        cv::cv2eigen(tvec, t_eig);
        current_frame_->setPose(SE3(R_eig, t_eig));
        
        // Update Frame Landmarks
        current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
        
        num_tracked_features_ = 0;
        for (int idx : inliers) {
            // idx is index in object_points/image_points
            int kp_idx = matched_kp_indices[idx];
            auto lm = matched_landmarks[idx];
            
            current_frame_->landmarks_[kp_idx] = lm;
            // Optionally update lm observation count etc.
            
            num_tracked_features_++;
        }
        
        return true;
    }

    std::cout << "TrackLocalMap: PnP failed. correspondences=" << object_points.size()
              << " used_global_fallback=" << (used_global_fallback ? 1 : 0) << std::endl;
    
    return false;
}

bool Tracking::trackReferenceKeyframe() {
    if (!last_frame_) return false;

    // Compute matches between current and last frame
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(current_frame_->descriptors_, last_frame_->descriptors_, knn, 2);

    struct MatchCandidate {
        int query_idx;
        int train_idx;
        float dist;
    };
    std::vector<MatchCandidate> candidates;
    candidates.reserve(knn.size());

    for (const auto& ms : knn) {
        if (ms.size() < 2) continue;
        const auto& m1 = ms[0];
        const auto& m2 = ms[1];
        if (m1.distance > 70.0f) continue;
        if (m1.distance >= 0.75f * m2.distance) continue;
        candidates.push_back({m1.queryIdx, m1.trainIdx, m1.distance});
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.dist < b.dist;
    });

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(candidates.size());
    std::vector<bool> used_query(current_frame_->keypoints_.size(), false);
    std::vector<bool> used_train(last_frame_->keypoints_.size(), false);
    for (const auto& c : candidates) {
        if (c.query_idx < 0 || c.query_idx >= static_cast<int>(current_frame_->keypoints_.size())) continue;
        if (c.train_idx < 0 || c.train_idx >= static_cast<int>(last_frame_->keypoints_.size())) continue;
        if (used_query[c.query_idx] || used_train[c.train_idx]) continue;
        used_query[c.query_idx] = true;
        used_train[c.train_idx] = true;
        good_matches.push_back(cv::DMatch(c.query_idx, c.train_idx, c.dist));
    }

    std::cout << "Matches with last frame: " << good_matches.size() << std::endl;

    if (good_matches.size() < 10) {
        return false;
    }
    
    // Propagate landmark associations from last frame via feature matches.
    // This is critical for bootstrapping 3D-2D PnP in subsequent frames.
    current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
    int propagated = 0;

    // Optimization: Pose from 3D-2D
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<int> match_indices; // index into good_matches for each 3D-2D pair
    
    for (const auto& m : good_matches) {
        // Query is current, Train is last
        int idx_last = m.trainIdx;
        int idx_curr = m.queryIdx;
        
        if (idx_last >= 0 && idx_last < static_cast<int>(last_frame_->landmarks_.size()) &&
            last_frame_->landmarks_[idx_last]) {
            // Found a map point
            Vec3 pos = last_frame_->landmarks_[idx_last]->getPos();
            object_points.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
            image_points.push_back(current_frame_->keypoints_[idx_curr].pt);

            current_frame_->landmarks_[idx_curr] = last_frame_->landmarks_[idx_last];
            match_indices.push_back(static_cast<int>(&m - &good_matches[0]));
            propagated++;
        }
    }

    if (propagated > 0) {
        std::cout << "TrackReferenceKeyframe: Propagated landmarks: " << propagated << std::endl;
    }

    std::cout << "TrackReferenceKeyframe: 3D-2D correspondences: " << object_points.size() << std::endl;
    
    if (object_points.size() >= 10) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        
        // Initial guess
        Eigen::Vector3d t = current_frame_->getPose().translation();
        Eigen::Matrix3d R = current_frame_->getPose().rotationMatrix();
        cv::Mat R_cv, t_cv;
        cv::eigen2cv(R, R_cv);
        cv::Rodrigues(R_cv, rvec);
        cv::eigen2cv(t, tvec);
        
        // Do not rely on initial guess here; allow RANSAC to find a pose even if motion model is poor.
        bool success = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                          rvec, tvec, false, 500, 20.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
                                          
        if (success) {
             std::cout << "TrackReferenceKeyframe: PnP Success, inliers: " << inliers.size() << std::endl;
             // Update pose
             cv::Mat R_new;
             cv::Rodrigues(rvec, R_new);
             Eigen::Matrix3d R_eig;
             Eigen::Vector3d t_eig;
             cv::cv2eigen(R_new, R_eig);
             cv::cv2eigen(tvec, t_eig);
             current_frame_->setPose(SE3(R_eig, t_eig));

             return true;
        }
    }

    // If PnP fails (e.g. no 3D points in last frame yet), we rely on motion model.
    // But since we are here, we probably have some tracking.
    
    return true;
}

}
