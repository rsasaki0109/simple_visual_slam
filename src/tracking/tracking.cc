#include "tracking/tracking.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <set>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "core/keyframe.h"
#include "core/landmark.h"
#include "sensors/accelerometer.h"

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

bool Tracking::initializeWithDepth() {
    // Single-frame initialization using depth map
    current_frame_->setPose(SE3());

    // Apply gravity alignment if accelerometer data is available
    if (!gravity_aligned_ && !accel_buffer_.empty()) {
        Vec3 gravity = AccelerometerProcessor::estimateGravity(accel_buffer_);
        if (gravity.norm() > 0.5) {
            Mat33 R_align = AccelerometerProcessor::computeGravityAlignment(gravity);
            SE3 T_aligned(R_align, Vec3(0, 0, 0));
            current_frame_->setPose(T_aligned);
            gravity_aligned_ = true;
            std::cout << "Tracking: Applied gravity alignment" << std::endl;
        }
    }

    auto kf = std::make_shared<Keyframe>(current_frame_);
    setKeyframeGravity(kf);

    // Back-project keypoints with valid depth to create landmarks
    int created = 0;
    static unsigned long depth_lm_id = 200000;
    SE3 T_wc = kf->T_cw_.inverse();

    for (size_t i = 0; i < kf->keypoints_.size(); ++i) {
        const auto& kp = kf->keypoints_[i];
        float depth = kf->getDepth(kp.pt.x, kp.pt.y);
        if (depth <= 0.0f || depth > 10.0f) continue;

        Vec3 p_norm = kf->camera_->unproject(Vec2(kp.pt.x, kp.pt.y));
        Vec3 p_cam = p_norm * static_cast<double>(depth);
        Vec3 p_w = T_wc * p_cam;

        auto lm = std::make_shared<Landmark>(depth_lm_id++, p_w);
        lm->addObservation(kf, i);
        lm->descriptor_ = kf->descriptors_.row(i).clone();

        kf->landmarks_[i] = lm;
        current_frame_->landmarks_[i] = lm;

        if (map_) map_->addLandmark(lm);
        created++;
    }

    if (created < 100) {
        std::cout << "Tracking: Depth init failed - only " << created << " points with valid depth" << std::endl;
        // Clean up
        if (map_) {
            for (auto& lm_pair : map_->getAllLandmarks()) {
                map_->removeLandmark(lm_pair.second);
            }
        }
        return false;
    }

    if (map_) {
        map_->addKeyframe(kf);
    }

    reference_keyframe_ = kf;
    initial_frame_ = current_frame_;
    state_ = TrackingState::OK;

    std::cout << "Tracking: Depth-based initialization SUCCESS! " << created
              << " 3D points from single frame (metric scale)" << std::endl;
    return true;
}

void Tracking::createLandmarksFromDepth(Keyframe::Ptr kf) {
    if (!kf || kf->depth_image_.empty()) return;

    SE3 T_wc = kf->T_cw_.inverse();
    int created = 0;
    static unsigned long depth_track_lm_id = 300000;

    for (size_t i = 0; i < kf->keypoints_.size(); ++i) {
        if (kf->landmarks_[i]) continue;  // Already has a landmark

        const auto& kp = kf->keypoints_[i];
        float depth = kf->getDepth(kp.pt.x, kp.pt.y);
        if (depth <= 0.0f || depth > 10.0f) continue;

        Vec3 p_norm = kf->camera_->unproject(Vec2(kp.pt.x, kp.pt.y));
        Vec3 p_cam = p_norm * static_cast<double>(depth);
        Vec3 p_w = T_wc * p_cam;

        auto lm = std::make_shared<Landmark>(depth_track_lm_id++, p_w);
        lm->addObservation(kf, i);
        lm->descriptor_ = kf->descriptors_.row(i).clone();

        kf->landmarks_[i] = lm;
        if (map_) map_->addLandmark(lm);
        created++;
    }

    if (created > 0) {
        std::cout << "Tracking: Created " << created << " landmarks from depth in KF " << kf->id_ << std::endl;
    }
}

bool Tracking::initialize() {
    if (!initial_frame_) {
        // First frame - try single-frame depth initialization
        if (!current_frame_->depth_image_.empty()) {
            initial_frame_ = current_frame_;
            if (initializeWithDepth()) {
                return true;
            }
            // Depth init failed, fall through to two-frame init
            initial_frame_ = nullptr;
        }

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
            setKeyframeGravity(kf_init);
            setKeyframeGravity(kf_cur);
            
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

            // Scale normalization: skip if we have metric depth data
            bool has_metric_depth = !current_frame_->depth_image_.empty() && current_frame_->depth_is_metric_;
            if (!has_metric_depth) {
                // Collect all depths for computing median
                std::vector<double> depths;
                for (auto& kv : map_->getAllLandmarks()) {
                    auto& lm = kv.second;
                    if (lm && !lm->isBad()) {
                        depths.push_back(lm->getPos().z());
                    }
                }
                if (!depths.empty()) {
                    std::sort(depths.begin(), depths.end());
                    double median_depth = depths[depths.size() / 2];
                    if (median_depth > 0.0) {
                        double scale = 1.0 / median_depth;
                        std::cout << "Tracking: Scaling map by " << scale << " (median depth was " << median_depth << ")" << std::endl;

                        for (auto& kv : map_->getAllLandmarks()) {
                            auto& lm = kv.second;
                            if (lm && !lm->isBad()) {
                                lm->setPos(lm->getPos() * scale);
                            }
                        }

                        SE3 T_scaled = SE3(kf_cur->T_cw_.so3(), kf_cur->T_cw_.translation() * scale);
                        kf_cur->T_cw_ = T_scaled;
                        current_frame_->setPose(T_scaled);
                    }
                }
            } else {
                std::cout << "Tracking: Metric depth available, skipping scale normalization" << std::endl;
            }

            if (map_) {
                map_->addKeyframe(kf_init);
                map_->addKeyframe(kf_cur);

                std::cout << "Tracking: Map after init: keyframes=" << map_->getAllKeyframes().size()
                          << " landmarks=" << map_->getAllLandmarks().size() << std::endl;
            }

            reference_keyframe_ = kf_cur;
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
    // If loop closing is correcting the map, use motion model only and skip all map access
    if (map_ && map_->loop_correcting_.load()) {
        if (last_frame_) {
            current_frame_->setPose(velocity_ * last_frame_->getPose());
        }
        return true;
    }

    // 1. Motion Model Prediction
    if (last_frame_) {
        // Check accelerometer for stationary detection
        bool stationary = false;
        if (!accel_buffer_.empty() && last_frame_) {
            auto recent_accel = std::vector<AccelEntry>();
            for (const auto& a : accel_buffer_) {
                if (a.timestamp_sec >= last_frame_->timestamp_ &&
                    a.timestamp_sec <= current_frame_->timestamp_) {
                    recent_accel.push_back(a);
                }
            }
            if (AccelerometerProcessor::isStationary(recent_accel)) {
                stationary = true;
            }
        }

        if (stationary) {
            // Zero motion prediction
            current_frame_->setPose(last_frame_->getPose());
        } else {
            // T_current = velocity * T_last
            SE3 T_cw_pred = velocity_ * last_frame_->getPose();
            current_frame_->setPose(T_cw_pred);
        }
    }

    // 2. Track Reference Keyframe (Frame-to-Frame matching for now)
    bool ref_tracking_ok = trackReferenceKeyframe();

    // 3. Track Local Map
    bool local_map_ok = false;
    if (ref_tracking_ok) {
        local_map_ok = trackLocalMap();
    }

    // 4. Handle tracking success/failure
    if (local_map_ok) {
        // Successful tracking
        state_ = TrackingState::OK;
        consecutive_tracking_failures_ = 0;
        lost_frame_count_ = 0;
        last_good_pose_ = current_frame_->getPose();

        // Update velocity for next frame (Constant velocity model)
        if (last_frame_) {
            velocity_ = current_frame_->getPose() * last_frame_->getPose().inverse();
        }
    } else {
        // Tracking failed - attempt relocalization
        std::cout << "Tracking: Lost, attempting relocalization..." << std::endl;

        if (relocalize()) {
            std::cout << "Tracking: Relocalization successful!" << std::endl;
            state_ = TrackingState::OK;
            consecutive_tracking_failures_ = 0;
            lost_frame_count_ = 0;
            velocity_ = SE3();  // Reset velocity after relocalization
        } else {
            // Relocalization failed
            state_ = TrackingState::LOST;
            consecutive_tracking_failures_++;
            lost_frame_count_++;

            if (consecutive_tracking_failures_ >= 3) {
                velocity_ = SE3();
            }

            // Keep trying relocalization rather than re-initializing
            // Re-initialization creates segments with inconsistent scale that degrades trajectory
            if (lost_frame_count_ > max_lost_frames_) {
                std::cout << "Tracking: Completely lost for " << lost_frame_count_ << " frames" << std::endl;
            }
        }
    }

    // 5. Check if we need a new Keyframe (only if tracking is OK)
    if (state_ == TrackingState::OK && needNewKeyframe()) {
        std::cout << "Tracking: Insert New Keyframe (ID=" << current_frame_->id_ << ")" << std::endl;
        // Create new Keyframe
        auto kf = std::make_shared<Keyframe>(current_frame_);
        setKeyframeGravity(kf);
        for (size_t i = 0; i < kf->landmarks_.size(); ++i) {
            auto& lm = kf->landmarks_[i];
            if (!lm || lm->isBad()) continue;
            lm->addObservation(kf, i);
        }

        // Create additional landmarks from depth for unmatched keypoints
        createLandmarksFromDepth(kf);

        // Update reference keyframe
        reference_keyframe_ = kf;

        if (local_mapping_) {
            local_mapping_->insertKeyframe(kf);
        } else {
            map_->addKeyframe(kf);
        }
    }

    return state_ == TrackingState::OK;
}

bool Tracking::needNewKeyframe() {
    if (!map_) return false;
    if (!reference_keyframe_) return false;

    // Heuristics for new keyframe decision:
    // 1. Min frames since last KF
    const int min_frames_since_last_kf = 3;
    if (current_frame_->id_ - reference_keyframe_->id_ < min_frames_since_last_kf) {
        return false;
    }

    // 2. Track quality: if tracked features drop below threshold, insert KF
    const int min_tracked_threshold = 60;
    if (num_tracked_features_ < min_tracked_threshold) {
        std::cout << "needNewKeyframe: Low tracked features (" << num_tracked_features_ << "), inserting KF." << std::endl;
        return true;
    }

    // 3. Max frames since last KF
    const int max_frames_since_last_kf = 12;
    if (current_frame_->id_ - reference_keyframe_->id_ >= max_frames_since_last_kf) {
        std::cout << "needNewKeyframe: Max frames reached, inserting KF." << std::endl;
        return true;
    }

    // 4. Ratio of tracked vs reference KF landmarks
    int ref_landmarks = 0;
    for (auto& lm : reference_keyframe_->landmarks_) {
        if (lm && !lm->isBad()) ref_landmarks++;
    }

    if (ref_landmarks > 0) {
        double ratio = static_cast<double>(num_tracked_features_) / ref_landmarks;
        if (ratio < 0.65) {
            std::cout << "needNewKeyframe: Low tracking ratio (" << ratio << "), inserting KF." << std::endl;
            return true;
        }
    }

    return false;
}

bool Tracking::trackLocalMap() {
    if (!map_) return false;

    // Skip if loop closing is correcting the map to avoid accessing invalidated landmarks
    if (map_->loop_correcting_.load()) {
        std::cout << "TrackLocalMap: Loop correction in progress, skipping" << std::endl;
        return true;
    }

    // 1. Build a true local map from the reference keyframe and its covisible neighbors.
    std::vector<Landmark::Ptr> landmarks;
    std::set<unsigned long> landmark_ids;

    auto add_landmarks_from_kf = [&](const Keyframe::Ptr& kf) {
        if (!kf) return;
        for (const auto& lm : kf->landmarks_) {
            if (!lm || lm->isBad()) continue;
            if (!landmark_ids.insert(lm->id_).second) continue;
            landmarks.push_back(lm);
        }
    };

    add_landmarks_from_kf(reference_keyframe_);
    if (reference_keyframe_) {
        const auto neighbors = reference_keyframe_->getBestCovisibilityKeyframes(15);
        for (const auto& neighbor : neighbors) {
            add_landmarks_from_kf(neighbor);
        }
    }

    if (landmarks.size() < 80) {
        const auto& all_landmarks = map_->getAllLandmarks();
        for (const auto& kv : all_landmarks) {
            const auto& lm = kv.second;
            if (!lm || lm->isBad()) continue;
            if (!landmark_ids.insert(lm->id_).second) continue;
            landmarks.push_back(lm);
        }
    }
    
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

    auto filter_correspondences_by_pose = [&](double base_gate_px) {
        if (object_points.empty()) return;

        std::vector<cv::Point3f> filtered_object_points;
        std::vector<cv::Point2f> filtered_image_points;
        std::vector<Landmark::Ptr> filtered_landmarks;
        std::vector<int> filtered_kp_indices;

        filtered_object_points.reserve(object_points.size());
        filtered_image_points.reserve(image_points.size());
        filtered_landmarks.reserve(matched_landmarks.size());
        filtered_kp_indices.reserve(matched_kp_indices.size());

        const SE3 T_cw_est = current_frame_->getPose();
        for (size_t i = 0; i < object_points.size(); ++i) {
            const auto& Pw = object_points[i];
            Vec3 p_w(Pw.x, Pw.y, Pw.z);
            Vec3 p_c = T_cw_est * p_w;
            if (!std::isfinite(p_c.x()) || !std::isfinite(p_c.y()) || !std::isfinite(p_c.z())) continue;
            if (p_c[2] <= 0.15 || p_c[2] > 18.0) continue;

            const int kp_idx = matched_kp_indices[i];
            const int octave = (kp_idx >= 0 && kp_idx < static_cast<int>(current_frame_->keypoints_.size()))
                ? current_frame_->keypoints_[kp_idx].octave
                : 0;
            const double gate_px = base_gate_px * (1.0 + 0.12 * static_cast<double>(std::max(0, octave)));

            Vec2 proj = current_frame_->camera_->project(p_c);
            const auto& uv = image_points[i];
            const double dx = uv.x - proj[0];
            const double dy = uv.y - proj[1];
            if ((dx * dx + dy * dy) > gate_px * gate_px) continue;

            filtered_object_points.push_back(Pw);
            filtered_image_points.push_back(uv);
            filtered_landmarks.push_back(matched_landmarks[i]);
            filtered_kp_indices.push_back(kp_idx);
        }

        object_points.swap(filtered_object_points);
        image_points.swap(filtered_image_points);
        matched_landmarks.swap(filtered_landmarks);
        matched_kp_indices.swap(filtered_kp_indices);
    };

    // 2. Project and Match
    int matches_found = 0;
    int visible_points = 0;
    int skipped_nonfinite = 0;
    int skipped_behind_or_close = 0;
    int skipped_oob = 0;
    for (const auto& lm : landmarks) {
        if (!lm) continue;
        if (lm->descriptor_.empty()) continue;
        
        // Project
        Vec3 pos_w = lm->getPos();
        if (!std::isfinite(pos_w[0]) || !std::isfinite(pos_w[1]) || !std::isfinite(pos_w[2])) {
            skipped_nonfinite++;
            continue;
        }
        Vec3 pos_c = current_frame_->getPose() * pos_w; // T_cw * pos_w

        // Check depth: must be positive and within reasonable range
        const double max_depth = 20.0;  // Reasonable for indoor scenes
        if (pos_c[2] <= 0.1 || pos_c[2] > max_depth) {
            skipped_behind_or_close++;
            continue; // Behind camera, too close, or too far
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
        double best_dist = 64.0;
        const double search_radius_sq = 120.0 * 120.0;
        
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

    filter_correspondences_by_pose(35.0);
    if (!object_points.empty()) {
        std::cout << "TrackLocalMap: Pose-gated matches: " << object_points.size() << std::endl;
    }

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

                // Stricter Lowe ratio test (0.6) + tighter absolute distance gate (50)
                if (m1.distance > 65.0f) continue;
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

            filter_correspondences_by_pose(55.0);

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

    auto refine_pnp_solution = [&](double refine_gate_px) -> bool {
        if (inliers.size() < 6) return false;

        std::vector<cv::Point3f> refine_object_points;
        std::vector<cv::Point2f> refine_image_points;
        std::vector<int> refine_indices;
        refine_object_points.reserve(inliers.size());
        refine_image_points.reserve(inliers.size());
        refine_indices.reserve(inliers.size());

        for (int idx : inliers) {
            if (idx < 0 || idx >= static_cast<int>(object_points.size())) continue;
            refine_object_points.push_back(object_points[idx]);
            refine_image_points.push_back(image_points[idx]);
            refine_indices.push_back(idx);
        }
        if (refine_object_points.size() < 6) return false;

        cv::Mat rvec_refined = rvec.clone();
        cv::Mat tvec_refined = tvec.clone();
        bool ok = cv::solvePnP(refine_object_points, refine_image_points, current_frame_->camera_->K(),
                               cv::Mat(), rvec_refined, tvec_refined, true, cv::SOLVEPNP_ITERATIVE);
        if (!ok) return false;

        std::vector<cv::Point2f> projected;
        cv::projectPoints(refine_object_points, rvec_refined, tvec_refined,
                          current_frame_->camera_->K(), cv::Mat(), projected);

        std::vector<cv::Point3f> gated_object_points;
        std::vector<cv::Point2f> gated_image_points;
        std::vector<int> gated_indices;
        gated_object_points.reserve(refine_object_points.size());
        gated_image_points.reserve(refine_image_points.size());
        gated_indices.reserve(refine_indices.size());

        for (size_t i = 0; i < projected.size(); ++i) {
            const int corr_idx = refine_indices[i];
            const int kp_idx = matched_kp_indices[corr_idx];
            const int octave = (kp_idx >= 0 && kp_idx < static_cast<int>(current_frame_->keypoints_.size()))
                ? current_frame_->keypoints_[kp_idx].octave
                : 0;
            const double gate_px = refine_gate_px * (1.0 + 0.10 * static_cast<double>(std::max(0, octave)));
            const double dx = projected[i].x - refine_image_points[i].x;
            const double dy = projected[i].y - refine_image_points[i].y;
            if ((dx * dx + dy * dy) > gate_px * gate_px) continue;
            gated_object_points.push_back(refine_object_points[i]);
            gated_image_points.push_back(refine_image_points[i]);
            gated_indices.push_back(corr_idx);
        }

        if (gated_object_points.size() < 6) return false;
        if (gated_object_points.size() != refine_object_points.size()) {
            rvec_refined = rvec.clone();
            tvec_refined = tvec.clone();
            ok = cv::solvePnP(gated_object_points, gated_image_points, current_frame_->camera_->K(),
                              cv::Mat(), rvec_refined, tvec_refined, true, cv::SOLVEPNP_ITERATIVE);
            if (!ok) return false;
        }

        rvec = rvec_refined;
        tvec = tvec_refined;
        inliers.swap(gated_indices);
        return true;
    };

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
                                     150, 10.0, 0.995, tmp_inliers, flag);
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
        success = refine_pnp_solution(8.0);
    }

    // Minimum inlier threshold for reliable pose estimation
    const size_t min_inliers = 15;

    if (success && inliers.size() >= min_inliers) {
        std::cout << "TrackLocalMap: PnP Success, inliers: " << inliers.size() << std::endl;
        // Update pose
        cv::Mat R_new;
        cv::Rodrigues(rvec, R_new);
        Eigen::Matrix3d R_eig;
        Eigen::Vector3d t_eig;
        cv::cv2eigen(R_new, R_eig);
        cv::cv2eigen(tvec, t_eig);
        SE3 new_pose(R_eig, t_eig);

        // Absolute pose sanity check: reject obviously wrong poses
        {
            Vec3 cam_pos = new_pose.inverse().translation();
            const double max_abs_pos = 50.0;  // Indoor scene: max 50m from origin
            if (std::abs(cam_pos.x()) > max_abs_pos || std::abs(cam_pos.y()) > max_abs_pos || std::abs(cam_pos.z()) > max_abs_pos) {
                std::cout << "TrackLocalMap: REJECTED - Absolute position out of bounds: " << cam_pos.transpose() << std::endl;
                return false;
            }
        }

        // Sanity check: reject poses with sudden large jumps
        if (last_frame_) {
            SE3 old_pose = last_frame_->getPose();
            Vec3 delta_t = new_pose.translation() - old_pose.translation();
            double trans_change = delta_t.norm();

            // Compute rotation change (angle in radians)
            // Sophus SO3 to rotation matrix, then AngleAxis
            Sophus::SO3d delta_rot = new_pose.so3().inverse() * old_pose.so3();
            Eigen::AngleAxisd aa(delta_rot.matrix());
            double rot_change = std::abs(aa.angle());

            // Thresholds:
            // - Translation: max 0.2 units per frame (strict for indoor)
            // - Rotation: max 0.3 radians (~17 degrees) per frame
            const double max_trans_change = 0.5;
            const double max_rot_change = 0.6;

            std::cout << "TrackLocalMap: Pose change - trans=" << trans_change
                      << " rot=" << rot_change << " rad" << std::endl;

            if (trans_change > max_trans_change || rot_change > max_rot_change) {
                std::cout << "TrackLocalMap: REJECTED - Pose change too large! "
                          << "trans=" << trans_change << " (max=" << max_trans_change << ") "
                          << "rot=" << rot_change << " (max=" << max_rot_change << ")" << std::endl;
                // Keep previous pose prediction (motion model)
                return false;
            }
        }

        current_frame_->setPose(new_pose);

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

    if (success && inliers.size() < min_inliers) {
        std::cout << "TrackLocalMap: PnP rejected - insufficient inliers: " << inliers.size()
                  << " (min=" << min_inliers << ")" << std::endl;
    } else {
        std::cout << "TrackLocalMap: PnP failed. correspondences=" << object_points.size()
                  << " used_global_fallback=" << (used_global_fallback ? 1 : 0) << std::endl;
    }

    return false;
}

bool Tracking::trackReferenceKeyframe() {
    if (!last_frame_) return false;
    if (map_ && map_->loop_correcting_.load()) return false;

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
        // Stricter Lowe ratio test (0.6) + tighter absolute distance gate (50)
        if (m1.distance > 65.0f) continue;
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

    if (good_matches.size() < 8) {
        return false;
    }
    
    // Propagate landmark associations from last frame via feature matches.
    // This is critical for bootstrapping 3D-2D PnP in subsequent frames.
    current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
    int propagated = 0;

    // Optimization: Pose from 3D-2D
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<int> current_kp_indices;
    std::vector<Landmark::Ptr> propagated_landmarks;
    
    for (const auto& m : good_matches) {
        // Query is current, Train is last
        int idx_last = m.trainIdx;
        int idx_curr = m.queryIdx;
        
        if (idx_last >= 0 && idx_last < static_cast<int>(last_frame_->landmarks_.size()) &&
            last_frame_->landmarks_[idx_last]) {
            // Found a map point
            Vec3 pos = last_frame_->landmarks_[idx_last]->getPos();
            if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) continue;

            Vec3 p_c = current_frame_->getPose() * pos;
            if (!std::isfinite(p_c.x()) || !std::isfinite(p_c.y()) || !std::isfinite(p_c.z())) continue;
            if (p_c[2] <= 0.15 || p_c[2] > 18.0) continue;

            const int octave = current_frame_->keypoints_[idx_curr].octave;
            const double gate_px = 48.0 * (1.0 + 0.12 * static_cast<double>(std::max(0, octave)));
            Vec2 proj = current_frame_->camera_->project(p_c);
            const auto& uv = current_frame_->keypoints_[idx_curr].pt;
            const double dx = uv.x - proj[0];
            const double dy = uv.y - proj[1];
            if ((dx * dx + dy * dy) > gate_px * gate_px) continue;

            object_points.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
            image_points.push_back(uv);

            current_frame_->landmarks_[idx_curr] = last_frame_->landmarks_[idx_last];
            current_kp_indices.push_back(idx_curr);
            propagated_landmarks.push_back(last_frame_->landmarks_[idx_last]);
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

        bool success = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                          rvec, tvec, true, 250, 8.0, 0.995, inliers, cv::SOLVEPNP_EPNP);

        auto refine_reference_pose = [&]() -> bool {
            if (inliers.size() < 6) return false;

            std::vector<cv::Point3f> refine_object_points;
            std::vector<cv::Point2f> refine_image_points;
            std::vector<int> refine_indices;
            refine_object_points.reserve(inliers.size());
            refine_image_points.reserve(inliers.size());
            refine_indices.reserve(inliers.size());

            for (int idx : inliers) {
                if (idx < 0 || idx >= static_cast<int>(object_points.size())) continue;
                refine_object_points.push_back(object_points[idx]);
                refine_image_points.push_back(image_points[idx]);
                refine_indices.push_back(idx);
            }
            if (refine_object_points.size() < 6) return false;

            cv::Mat rvec_refined = rvec.clone();
            cv::Mat tvec_refined = tvec.clone();
            bool ok = cv::solvePnP(refine_object_points, refine_image_points, current_frame_->camera_->K(),
                                   cv::Mat(), rvec_refined, tvec_refined, true, cv::SOLVEPNP_ITERATIVE);
            if (!ok) return false;

            std::vector<cv::Point2f> projected;
            cv::projectPoints(refine_object_points, rvec_refined, tvec_refined,
                              current_frame_->camera_->K(), cv::Mat(), projected);

            std::vector<cv::Point3f> gated_object_points;
            std::vector<cv::Point2f> gated_image_points;
            std::vector<int> gated_indices;
            for (size_t i = 0; i < projected.size(); ++i) {
                const int corr_idx = refine_indices[i];
                const int kp_idx = current_kp_indices[corr_idx];
                const int octave = current_frame_->keypoints_[kp_idx].octave;
                const double gate_px = 6.0 * (1.0 + 0.10 * static_cast<double>(std::max(0, octave)));
                const double dx = projected[i].x - refine_image_points[i].x;
                const double dy = projected[i].y - refine_image_points[i].y;
                if ((dx * dx + dy * dy) > gate_px * gate_px) continue;
                gated_object_points.push_back(refine_object_points[i]);
                gated_image_points.push_back(refine_image_points[i]);
                gated_indices.push_back(corr_idx);
            }

            if (gated_object_points.size() < 6) return false;
            if (gated_object_points.size() != refine_object_points.size()) {
                rvec_refined = rvec.clone();
                tvec_refined = tvec.clone();
                ok = cv::solvePnP(gated_object_points, gated_image_points, current_frame_->camera_->K(),
                                  cv::Mat(), rvec_refined, tvec_refined, true, cv::SOLVEPNP_ITERATIVE);
                if (!ok) return false;
            }

            rvec = rvec_refined;
            tvec = tvec_refined;
            inliers.swap(gated_indices);
            return true;
        };

        if (success) {
            success = refine_reference_pose();
        }
                                          
        // Minimum inlier threshold
        const size_t min_inliers = 15;

        if (success && inliers.size() >= min_inliers) {
             std::cout << "TrackReferenceKeyframe: PnP Success, inliers: " << inliers.size() << std::endl;
             // Update pose
             cv::Mat R_new;
             cv::Rodrigues(rvec, R_new);
             Eigen::Matrix3d R_eig;
             Eigen::Vector3d t_eig;
             cv::cv2eigen(R_new, R_eig);
             cv::cv2eigen(tvec, t_eig);
             SE3 new_pose(R_eig, t_eig);

             // Absolute pose sanity check
             {
                 Vec3 cam_pos = new_pose.inverse().translation();
                 const double max_abs_pos = 50.0;
                 if (std::abs(cam_pos.x()) > max_abs_pos || std::abs(cam_pos.y()) > max_abs_pos || std::abs(cam_pos.z()) > max_abs_pos) {
                     std::cout << "TrackReferenceKeyframe: REJECTED - Absolute position out of bounds: " << cam_pos.transpose() << std::endl;
                     return false;
                 }
             }

             // Sanity check: reject poses with sudden large jumps
             if (last_frame_) {
                 SE3 old_pose = last_frame_->getPose();
                 Vec3 delta_t = new_pose.translation() - old_pose.translation();
                 double trans_change = delta_t.norm();

                 Sophus::SO3d delta_rot = new_pose.so3().inverse() * old_pose.so3();
                 Eigen::AngleAxisd aa(delta_rot.matrix());
                 double rot_change = std::abs(aa.angle());

                 const double max_trans_change = 0.35;
                 const double max_rot_change = 0.45;

                 std::cout << "TrackReferenceKeyframe: Pose change - trans=" << trans_change
                           << " rot=" << rot_change << " rad" << std::endl;

                 if (trans_change > max_trans_change || rot_change > max_rot_change) {
                     std::cout << "TrackReferenceKeyframe: REJECTED - Pose change too large!" << std::endl;
                     // Fall through to use motion model
                 } else {
                     current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
                     for (int idx : inliers) {
                         if (idx < 0 || idx >= static_cast<int>(current_kp_indices.size())) continue;
                         current_frame_->landmarks_[current_kp_indices[idx]] = propagated_landmarks[idx];
                     }
                     current_frame_->setPose(new_pose);
                     return true;
                 }
             } else {
                 current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
                 for (int idx : inliers) {
                     if (idx < 0 || idx >= static_cast<int>(current_kp_indices.size())) continue;
                     current_frame_->landmarks_[current_kp_indices[idx]] = propagated_landmarks[idx];
                 }
                 current_frame_->setPose(new_pose);
                 return true;
             }
        }
    }

    // If PnP fails (e.g. no 3D points in last frame yet), we rely on motion model.
    // But since we are here, we probably have some tracking.

    return true;
}

void Tracking::onBACompleted() {
    std::lock_guard<std::mutex> lock(pose_mutex_);

    if (!current_frame_) {
        return;
    }

    // Skip if current frame is a keyframe (its pose was optimized in BA)
    // Check by comparing with reference keyframe
    if (reference_keyframe_ && current_frame_->id_ == reference_keyframe_->id_) {
        return;
    }

    std::cout << "Tracking: BA completed, recomputing current frame pose..." << std::endl;
    recomputeCurrentPose();
}

void Tracking::recomputeCurrentPose() {
    // Recompute current frame pose using updated landmark positions from BA
    if (!current_frame_ || !current_frame_->camera_) {
        return;
    }

    // Collect 3D-2D correspondences from current frame's landmark associations
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> indices;

    for (size_t i = 0; i < current_frame_->landmarks_.size(); i++) {
        auto lm = current_frame_->landmarks_[i];
        if (!lm || lm->isBad()) continue;

        Vec3 pos = lm->getPos();  // BA-updated position
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
            continue;
        }

        pts3d.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
        pts2d.push_back(current_frame_->keypoints_[i].pt);
        indices.push_back(static_cast<int>(i));
    }

    std::cout << "Tracking::recomputeCurrentPose: 3D-2D pairs = " << pts3d.size() << std::endl;

    if (pts3d.size() < 10) {
        std::cout << "Tracking::recomputeCurrentPose: Not enough correspondences" << std::endl;
        return;
    }

    // PnP to recompute pose
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(pts3d, pts2d, current_frame_->camera_->K(), cv::Mat(),
                                  rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

    if (ok && inliers.size() >= 20) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d R_eig;
        Eigen::Vector3d t_eig;
        cv::cv2eigen(R, R_eig);
        cv::cv2eigen(tvec, t_eig);

        SE3 new_pose(R_eig, t_eig);
        SE3 old_pose = current_frame_->getPose();

        // Compute reprojection error before and after
        double err_before = 0.0, err_after = 0.0;
        int valid_count = 0;
        for (int idx : inliers) {
            const auto& P = pts3d[idx];
            const auto& uv = pts2d[idx];
            Vec3 p_w(P.x, P.y, P.z);

            // Before (old pose)
            Vec3 p_c_old = old_pose * p_w;
            if (p_c_old.z() > 0) {
                Vec2 proj_old = current_frame_->camera_->project(p_c_old);
                err_before += std::sqrt((uv.x - proj_old[0]) * (uv.x - proj_old[0]) +
                                        (uv.y - proj_old[1]) * (uv.y - proj_old[1]));
            }

            // After (new pose)
            Vec3 p_c_new = new_pose * p_w;
            if (p_c_new.z() > 0) {
                Vec2 proj_new = current_frame_->camera_->project(p_c_new);
                err_after += std::sqrt((uv.x - proj_new[0]) * (uv.x - proj_new[0]) +
                                       (uv.y - proj_new[1]) * (uv.y - proj_new[1]));
                valid_count++;
            }
        }

        if (valid_count > 0) {
            err_before /= valid_count;
            err_after /= valid_count;
        }

        std::cout << "Tracking::recomputeCurrentPose: Reprojection error before=" << err_before
                  << " after=" << err_after << " inliers=" << inliers.size() << std::endl;

        // Only update if new pose is better (lower reprojection error)
        if (err_after < err_before || err_after < 20.0) {
            current_frame_->setPose(new_pose);

            // Also update velocity model
            if (last_frame_) {
                velocity_ = current_frame_->getPose() * last_frame_->getPose().inverse();
            }

            std::cout << "Tracking::recomputeCurrentPose: Pose updated successfully" << std::endl;
        } else {
            std::cout << "Tracking::recomputeCurrentPose: New pose rejected (higher error)" << std::endl;
        }
    } else {
        std::cout << "Tracking::recomputeCurrentPose: PnP failed or insufficient inliers ("
                  << inliers.size() << ")" << std::endl;
    }
}

bool Tracking::relocalize() {
    if (!map_) return false;

    // Get all keyframes from map
    auto keyframes = map_->getAllKeyframes();
    if (keyframes.empty()) {
        std::cout << "Relocalize: No keyframes available" << std::endl;
        return false;
    }

    const cv::Mat& curr_desc = current_frame_->descriptors_;
    if (curr_desc.empty()) {
        std::cout << "Relocalize: Current frame has no descriptors" << std::endl;
        return false;
    }

    // Structure to hold candidate keyframes with their match scores
    struct Candidate {
        Keyframe::Ptr kf;
        std::vector<cv::DMatch> matches;
        int score;
    };
    std::vector<Candidate> candidates;

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Match current frame with each keyframe
    for (auto& kf_pair : keyframes) {
        auto& kf = kf_pair.second;
        if (!kf || kf->descriptors_.empty()) continue;

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(curr_desc, kf->descriptors_, knn, 2);

        std::vector<cv::DMatch> good;
        for (auto& m : knn) {
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance && m[0].distance < 65.0f) {
                good.push_back(m[0]);
            }
        }

        if (good.size() >= 8) {  // Slightly lower minimum match threshold for recovery
            candidates.push_back({kf, good, static_cast<int>(good.size())});
        }
    }

    if (candidates.empty()) {
        std::cout << "Relocalize: No candidate keyframes found" << std::endl;
        return false;
    }

    // Sort by score (number of matches)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.score > b.score;
              });

    std::cout << "Relocalize: Found " << candidates.size() << " candidate keyframes" << std::endl;

    // Try PnP with top N candidates
    const int max_candidates = 10;
    for (int i = 0; i < std::min(static_cast<int>(candidates.size()), max_candidates); i++) {
        auto& cand = candidates[i];

        // Build 3D-2D correspondences
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        std::vector<int> match_indices;

        for (size_t m_idx = 0; m_idx < cand.matches.size(); m_idx++) {
            auto& m = cand.matches[m_idx];
            int kf_idx = m.trainIdx;
            int curr_idx = m.queryIdx;

            if (kf_idx >= 0 && kf_idx < static_cast<int>(cand.kf->landmarks_.size()) &&
                cand.kf->landmarks_[kf_idx]) {
                auto lm = cand.kf->landmarks_[kf_idx];
                if (lm->isBad()) continue;

                Vec3 pos = lm->getPos();
                if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
                    continue;
                }

                pts3d.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
                pts2d.push_back(current_frame_->keypoints_[curr_idx].pt);
                match_indices.push_back(static_cast<int>(m_idx));
            }
        }

        if (pts3d.size() < 10) {
            continue;
        }

        // PnP with RANSAC
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(pts3d, pts2d, current_frame_->camera_->K(), cv::Mat(),
                                      rvec, tvec, false, 500, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

        if (ok && inliers.size() >= 12) {  // Lower threshold for relocalization
            // Success! Validate pose before accepting
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            Eigen::Matrix3d R_eig;
            Eigen::Vector3d t_eig;
            cv::cv2eigen(R, R_eig);
            cv::cv2eigen(tvec, t_eig);

            SE3 candidate_pose(R_eig, t_eig);
            // Sanity check: camera position should be near the candidate KF
            SE3 T_wc_cand = cand.kf->T_cw_.inverse();
            SE3 T_wc_new = candidate_pose.inverse();
            double dist_to_kf = (T_wc_new.translation() - T_wc_cand.translation()).norm();
            if (dist_to_kf > 5.0) {
                std::cout << "Relocalize: Rejected KF " << cand.kf->id_
                          << " - pose too far from KF (" << dist_to_kf << "m)" << std::endl;
                continue;
            }

            current_frame_->setPose(candidate_pose);

            // Set landmark associations for inliers
            current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
            for (int idx : inliers) {
                int orig_match_idx = match_indices[idx];
                int kf_idx = cand.matches[orig_match_idx].trainIdx;
                int curr_idx = cand.matches[orig_match_idx].queryIdx;
                if (kf_idx >= 0 && kf_idx < static_cast<int>(cand.kf->landmarks_.size())) {
                    current_frame_->landmarks_[curr_idx] = cand.kf->landmarks_[kf_idx];
                }
            }

            // Update reference keyframe
            reference_keyframe_ = cand.kf;

            std::cout << "Relocalize: Matched with KF " << cand.kf->id_
                      << " inliers=" << inliers.size() << std::endl;
            return true;
        }
    }

    std::cout << "Relocalize: All candidate PnP attempts failed" << std::endl;
    return false;
}

bool Tracking::reinitialize() {
    if (!current_frame_) return false;

    // First call after entering re-init mode: store reference frame
    if (!reinit_reference_frame_) {
        reinit_reference_frame_ = current_frame_;
        reinit_initializer_ = std::make_shared<Initializer>(current_frame_);
        std::cout << "Tracking: Re-init reference set (frame " << current_frame_->id_ << ")" << std::endl;
        return false;
    }

    // Check if enough disparity has accumulated since reference
    // (avoid trying to initialize from nearly identical frames)
    if (current_frame_->id_ - reinit_reference_frame_->id_ < 3) {
        return false;
    }

    // Try initialization
    bool ok = reinit_initializer_->initialize(current_frame_);
    if (!ok) {
        // If too many frames without success, reset reference
        if (current_frame_->id_ - reinit_reference_frame_->id_ > 30) {
            std::cout << "Tracking: Re-init timeout, resetting reference frame" << std::endl;
            reinit_reference_frame_ = current_frame_;
            reinit_initializer_ = std::make_shared<Initializer>(current_frame_);
        }
        return false;
    }

    // Initialization succeeded!
    std::cout << "Tracking: Re-init triangulation starting..." << std::endl;

    auto kf_ref = std::make_shared<Keyframe>(reinit_reference_frame_);
    setKeyframeGravity(kf_ref);
    auto kf_cur = std::make_shared<Keyframe>(current_frame_);
    setKeyframeGravity(kf_cur);

    // Set poses - anchor the new segment to the last known good pose
    // This keeps the new segment in the same coordinate system as the existing map
    SE3 T_anchor = last_good_pose_;  // Last known good T_cw before tracking loss
    reinit_reference_frame_->setPose(T_anchor);
    kf_ref->T_cw_ = T_anchor;

    // Poses will be set after scale normalization below

    // Triangulate points in local frame first, then normalize scale, then transform to world
    struct TriPoint {
        Vec3 pos_local;
        int idx_ref;
        int idx_cur;
    };
    std::vector<TriPoint> tri_points;
    std::vector<double> depths;

    for (size_t i = 0; i < reinit_initializer_->is_triangulated_.size(); ++i) {
        if (!reinit_initializer_->is_triangulated_[i]) continue;

        cv::Point3f pt3d = reinit_initializer_->triangulated_points_[i];
        if (!std::isfinite(pt3d.x) || !std::isfinite(pt3d.y) || !std::isfinite(pt3d.z)) continue;
        if (pt3d.z <= 0.0f) continue;
        if (std::abs(pt3d.x) > 1e4f || std::abs(pt3d.y) > 1e4f || std::abs(pt3d.z) > 1e4f) continue;

        depths.push_back(pt3d.z);
        tri_points.push_back({
            Vec3(pt3d.x, pt3d.y, pt3d.z),
            reinit_initializer_->matches_[i].queryIdx,
            reinit_initializer_->matches_[i].trainIdx
        });
    }

    if (tri_points.size() < 50) {
        std::cout << "Tracking: Re-init failed - only " << tri_points.size() << " points triangulated" << std::endl;
        reinit_reference_frame_ = current_frame_;
        reinit_initializer_ = std::make_shared<Initializer>(current_frame_);
        return false;
    }

    // Normalize scale in local frame using median depth
    // This sets median depth to 1.0m (reasonable for indoor scenes)
    double scale = 1.0;
    if (!depths.empty()) {
        std::sort(depths.begin(), depths.end());
        double median_depth = depths[depths.size() / 2];
        if (median_depth > 0.0) {
            scale = 1.0 / median_depth;
        }
    }

    // Scale the relative pose T_c1_c2 in local frame, then compose with anchor
    SE3 T_c1_c2_scaled = SE3(
        reinit_initializer_->T_c1_c2_.so3(),
        reinit_initializer_->T_c1_c2_.translation() * scale);
    SE3 T_cur = T_c1_c2_scaled * T_anchor;
    kf_cur->T_cw_ = T_cur;
    current_frame_->setPose(T_cur);

    // Transform scaled points from local frame to world frame
    SE3 T_wc_anchor = T_anchor.inverse();
    size_t inserted = 0;

    for (auto& tp : tri_points) {
        Vec3 pos_local_scaled = tp.pos_local * scale;
        Vec3 pos_w = T_wc_anchor * pos_local_scaled;

        static unsigned long reinit_lm_id = 100000;
        auto lm = std::make_shared<Landmark>(reinit_lm_id++, pos_w);

        lm->addObservation(kf_ref, tp.idx_ref);
        lm->addObservation(kf_cur, tp.idx_cur);
        lm->descriptor_ = reinit_reference_frame_->descriptors_.row(tp.idx_ref).clone();

        kf_ref->landmarks_[tp.idx_ref] = lm;
        kf_cur->landmarks_[tp.idx_cur] = lm;
        reinit_reference_frame_->landmarks_[tp.idx_ref] = lm;
        current_frame_->landmarks_[tp.idx_cur] = lm;

        if (map_) {
            map_->addLandmark(lm);
            inserted++;
        }
    }

    // Add keyframes to map
    if (map_) {
        map_->addKeyframe(kf_ref);
        map_->addKeyframe(kf_cur);
    }

    reference_keyframe_ = kf_cur;

    // Reset re-init state
    reinit_reference_frame_ = nullptr;
    reinit_initializer_ = nullptr;

    std::cout << "Tracking: Re-init complete! " << inserted << " landmarks, "
              << map_->getAllKeyframes().size() << " total KFs" << std::endl;
    return true;
}

void Tracking::setKeyframeGravity(Keyframe::Ptr kf) {
    if (!kf || !gravity_aligned_ || accel_buffer_.empty()) return;

    // Find accelerometer readings near keyframe timestamp (±50ms window)
    double ts = kf->timestamp_;
    double window = 0.05;
    std::vector<AccelEntry> nearby;
    for (const auto& a : accel_buffer_) {
        if (a.timestamp_sec >= ts - window && a.timestamp_sec <= ts + window) {
            nearby.push_back(a);
        }
        if (a.timestamp_sec > ts + window) break;
    }

    if (nearby.size() < 3) return;
    // Only apply gravity prior during low-dynamic-acceleration periods
    // High threshold (5.0) to include most frames except high-acceleration moments
    if (!AccelerometerProcessor::isStationary(nearby, 5.0)) return;

    // Compute gravity direction in sensor frame, then transform to camera frame
    Vec3 g_sensor = AccelerometerProcessor::estimateGravity(nearby);
    if (g_sensor.norm() < 0.5) return;

    // For TUM datasets, accelerometer frame ≈ camera frame (close enough for prior)
    kf->gravity_in_camera_ = g_sensor.normalized();
    kf->has_gravity_ = true;
}

}
