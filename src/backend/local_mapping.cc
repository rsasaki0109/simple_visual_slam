#include "backend/local_mapping.h"
#include "core/landmark.h"
#include "core/keyframe.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

namespace svslam {

LocalMapping::LocalMapping(Map::Ptr map) : map_(map) {}

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
            
            // optimization(); // TODO: Implement Local BA
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
    
    std::cout << "LocalMapping: Processed Keyframe " << current_processed_kf_->id_ 
              << ". Connected KFs: " << current_processed_kf_->connected_keyframes_.size() << std::endl;
}

void LocalMapping::mapPointCulling() {
    // Remove bad map points
    // For now, stub
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
    // Stub
}

}
