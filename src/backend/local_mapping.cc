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
        
        // Process all available keyframes (or maybe just one by one?)
        if (checkNewKeyframes()) {
            processNewKeyframe();
            mapPointCulling();
            createNewMapPoints();
            
            optimization(); 
        }
    }
    std::cout << "LocalMapping thread stopped." << std::endl;
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
    
    // Pass to Loop Closing
    if (loop_closing_) {
        loop_closing_->insertKeyframe(current_processed_kf_);
    }
    
    std::cout << "LocalMapping: Processed Keyframe " << current_processed_kf_->id_ 
              << ". Connected KFs: " << current_processed_kf_->connected_keyframes_.size() << std::endl;
}

void LocalMapping::mapPointCulling() {
    // Remove bad map points
    // 1. Check recent map points
    auto recent_landmarks = map_->getAllLandmarks(); // This gets ALL, inefficient. 
    // Ideally we should track "recently created" landmarks.
    
    // For now, let's just scan landmarks connected to current KF.
    std::vector<Landmark::Ptr> landmarks_to_check;
    for (auto& lm : current_processed_kf_->landmarks_) {
        if (lm && !lm->isBad()) {
            landmarks_to_check.push_back(lm);
        }
    }
    
    int culled = 0;
    for (auto& lm : landmarks_to_check) {
        if (lm->isBad()) continue;
        
        // Policy:
        // 1. If observation count < 2 (only created and never seen again?)
        //    Maybe give it a grace period (e.g. by frame ID or creation time)
        // 2. Or if ratio of found/visible is low (we don't track visible count yet)
        
        if (lm->observations_.size() < 2) {
            // If it was created more than 2 frames ago and still has < 2 obs, remove it
            // We need creation time in Landmark. Assuming ID correlates with time.
            
            // Stub: just remove strict ones for now to keep map clean?
            // Actually, newly created points will have 2 obs (Current + Neighbor).
            // So < 2 shouldn't happen unless connection broken.
            
            // Let's cull if observations == 1 (orphaned)
            if (lm->observations_.size() == 1) {
                // Determine if we should delete
                // auto obs = lm->observations_.begin();
                // auto kf = obs->first.lock();
                
                // For simplicity in this mono slam:
                lm->setBad();
                map_->removeLandmark(lm);
                culled++;
            }
        }
    }
    
    if (culled > 0)
        std::cout << "LocalMapping: Culled " << culled << " map points." << std::endl;
}

void LocalMapping::createNewMapPoints() {
    // Triangulate new points between current_processed_kf_ and its neighbors
    int nn = 10;
    std::vector<Keyframe::Ptr> neighbors = current_processed_kf_->getBestCovisibilityKeyframes(nn);
    
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
    
    for (auto& neighbor : neighbors) {
        if (!neighbor) continue;
        SE3 T_cw2 = neighbor->T_cw_;
        Eigen::Vector3d Ow2 = T_cw2.so3().inverse() * -T_cw2.translation();
        
        // Baseline check
        double baseline = (Ow1 - Ow2).norm();
        if (baseline < 0.05) continue; // Too small baseline
        
        std::vector<int> unmatched_indices_2;
        cv::Mat descriptors_2;
        for (size_t i=0; i < neighbor->keypoints_.size(); ++i) {
            if (!neighbor->landmarks_[i]) {
                unmatched_indices_2.push_back(i);
                descriptors_2.push_back(neighbor->descriptors_.row(i));
            }
        }
        
        if (descriptors_2.empty()) continue;
        
        std::vector<cv::DMatch> matches;
        matcher->match(descriptors_1, descriptors_2, matches);
        
        for (auto& m : matches) {
            if (m.distance > 50) continue;
            
            int idx1 = unmatched_indices_1[m.queryIdx];
            int idx2 = unmatched_indices_2[m.trainIdx];
            
            // Triangulate
            std::vector<cv::Point2f> pts1, pts2;
            pts1.push_back(current_processed_kf_->keypoints_[idx1].pt);
            pts2.push_back(neighbor->keypoints_[idx2].pt);
            
            cv::Mat pt_4d;
            
            // Convert Pose to OpenCV Mat
            // T_cw is 3x4 [R|t] or 4x4
            // matrix3x4() returns Eigen matrix
            Eigen::Matrix<double, 3, 4> mat1 = T_cw1.matrix3x4();
            Eigen::Matrix<double, 3, 4> mat2 = T_cw2.matrix3x4();
            cv::Mat T1_cv, T2_cv;
            cv::eigen2cv(mat1, T1_cv);
            cv::eigen2cv(mat2, T2_cv);
            
            // K * T_cw
            cv::Mat P1 = current_processed_kf_->camera_->K() * T1_cv;
            cv::Mat P2 = neighbor->camera_->K() * T2_cv;
            
            cv::triangulatePoints(P1, P2, pts1, pts2, pt_4d);
            
            // Check parallax and depth
            // pt_4d is 4xN. Since we passed vectors of size 1, it's 4x1.
            // Using double because K is double.
            
            if (pt_4d.at<double>(3, 0) == 0) continue;
            
            cv::Point3f pt_w(pt_4d.at<double>(0, 0) / pt_4d.at<double>(3, 0),
                             pt_4d.at<double>(1, 0) / pt_4d.at<double>(3, 0),
                             pt_4d.at<double>(2, 0) / pt_4d.at<double>(3, 0));
                             
            // Simple depth check
            Vec3 P(pt_w.x, pt_w.y, pt_w.z);
            double d1 = (T_cw1 * P)[2];
            double d2 = (T_cw2 * P)[2];
            
            if (d1 > 0 && d2 > 0) {
                 // Success - create map point
                 static unsigned long lm_id_counter = 10000; // TODO: Global ID
                 auto lm = std::make_shared<Landmark>(lm_id_counter++, P);
                 
                 lm->addObservation(current_processed_kf_, idx1);
                 lm->addObservation(neighbor, idx2);
                 lm->descriptor_ = current_processed_kf_->descriptors_.row(idx1).clone();
                 
                 // Add to keyframes
                 current_processed_kf_->landmarks_[idx1] = lm;
                 neighbor->landmarks_[idx2] = lm;
                 
                 map_->addLandmark(lm);
            }
        }
    }
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
    std::set<Landmark::Ptr> local_landmarks_set;
    for (auto& kf : local_keyframes) {
        for (auto& lm : kf->landmarks_) {
            if (lm && !lm->isBad()) {
                local_landmarks_set.insert(lm);
            }
        }
    }
    std::vector<Landmark::Ptr> local_landmarks(local_landmarks_set.begin(), local_landmarks_set.end());
    
    std::cout << "LocalMapping: BA on " << local_keyframes.size() << " KFs and " << local_landmarks.size() << " LMs." << std::endl;
    
    if (local_keyframes.size() < 2 || local_landmarks.size() < 10) return;
    
    Optimizer::bundleAdjustment(local_keyframes, local_landmarks, 5);
}

}
