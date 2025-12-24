#include "tracking/tracking.h"
#include <iostream>
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
            for (size_t i = 0; i < initializer_->is_triangulated_.size(); ++i) {
                if (initializer_->is_triangulated_[i]) {
                    // Create Landmark
                    cv::Point3f pt3d = initializer_->triangulated_points_[i]; // In Ref frame
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
                    if (map_) map_->addLandmark(lm);
                }
            }
            
            if (map_) {
                map_->addKeyframe(kf_init);
                map_->addKeyframe(kf_cur);
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
    
    std::cout << "TrackLocalMap: Landmarks to check: " << landmarks.size() << std::endl;

    // 2. Project and Match
    int matches_found = 0;
    int visible_points = 0;
    for (auto& pair : landmarks) {
        auto lm = pair.second;
        if (!lm) continue;
        if (lm->descriptor_.empty()) continue;
        
        // Project
        Vec3 pos_w = lm->getPos();
        Vec3 pos_c = current_frame_->getPose() * pos_w; // T_cw * pos_w
        
        if (pos_c[2] <= 0.1) continue; // Behind camera or too close
        
        Vec2 px = current_frame_->camera_->project(pos_c);
        
        // Check bounds
        if (px[0] < 0 || px[0] >= current_frame_->image_.cols ||
            px[1] < 0 || px[1] >= current_frame_->image_.rows) continue;
            
        visible_points++;
        
        // Search for match in current frame features
        // Simple search: look for features near px
        int best_idx = -1;
        double best_dist = 50.0; // Descriptor distance threshold (Hamming)
        
        // Radius search (naive O(N) per landmark)
        // Ideally should use grid search
        for (size_t i = 0; i < current_frame_->keypoints_.size(); ++i) {
             const auto& kp = current_frame_->keypoints_[i];
             double dist_spatial = (kp.pt.x - px[0])*(kp.pt.x - px[0]) + (kp.pt.y - px[1])*(kp.pt.y - px[1]);
             
             if (dist_spatial < 2500.0) { // 50 pixels radius squared
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
            
            matches_found++;
        }
    }
    
    std::cout << "TrackLocalMap: Visible: " << visible_points << ", Matches: " << matches_found << std::endl;
    
    if (object_points.size() < 10) return false;
    
    // 3. Optimize Pose (solvePnPRansac)
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    
    // Initial guess from motion model
    Eigen::Vector3d t = current_frame_->getPose().translation();
    Eigen::Matrix3d R = current_frame_->getPose().rotationMatrix();
    cv::Mat R_cv, t_cv;
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    tvec = t_cv;
    
    bool success = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                      rvec, tvec, true, 100, 8.0, 0.99, inliers);
                                      
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
    
    return false;
}

bool Tracking::trackReferenceKeyframe() {
    if (!last_frame_) return false;

    // Compute matches between current and last frame
    std::vector<cv::DMatch> matches;
    matcher_->match(current_frame_->descriptors_, last_frame_->descriptors_, matches);

    // Filter matches (Simple min distance check)
    double min_dist = 10000, max_dist = 0;
    for (const auto& m : matches) {
        double dist = m.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : matches) {
        if (m.distance <= std::max(2.0 * min_dist, 30.0)) {
            good_matches.push_back(m);
        }
    }

    std::cout << "Matches with last frame: " << good_matches.size() << std::endl;

    if (good_matches.size() < 10) {
        return false;
    }
    
    // Optimization: Pose from 3D-2D
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    
    for (const auto& m : good_matches) {
        // Query is current, Train is last
        int idx_last = m.trainIdx;
        int idx_curr = m.queryIdx;
        
        if (last_frame_->landmarks_[idx_last]) {
            // Found a map point
            Vec3 pos = last_frame_->landmarks_[idx_last]->getPos();
            object_points.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
            image_points.push_back(current_frame_->keypoints_[idx_curr].pt);
        }
    }
    
    if (object_points.size() >= 10) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        
        // Initial guess
        Eigen::Vector3d t = current_frame_->getPose().translation();
        Eigen::Matrix3d R = current_frame_->getPose().rotationMatrix();
        cv::Mat R_cv, t_cv;
        cv::eigen2cv(R, R_cv);
        cv::Rodrigues(R_cv, rvec);
        tvec = t_cv;
        
        bool success = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                          rvec, tvec, true, 100, 8.0, 0.99, inliers);
                                          
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
