#include "tracking/initializer.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace svslam {

Initializer::Initializer(Frame::Ptr frame_ref) : frame_ref_(frame_ref) {}

bool Initializer::initialize(Frame::Ptr frame_cur, float sigma, int max_iterations) {
    // 1. Match features
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<cv::DMatch> all_matches;
    matcher->match(frame_ref_->descriptors_, frame_cur->descriptors_, all_matches);

    // Filter matches
    double min_dist = 10000;
    for (const auto& m : all_matches) {
        if (m.distance < min_dist) min_dist = m.distance;
    }

    matches_.clear();
    kps_ref_.clear();
    kps_cur_.clear();
    for (const auto& m : all_matches) {
        if (m.distance <= std::max(2.0 * min_dist, 30.0)) {
            matches_.push_back(m);
            kps_ref_.push_back(frame_ref_->keypoints_[m.queryIdx].pt);
            kps_cur_.push_back(frame_cur->keypoints_[m.trainIdx].pt);
        }
    }

    std::cout << "Initializer: Matches found: " << matches_.size() << std::endl;
    if (matches_.size() < 100) {
        std::cout << "Initializer: Not enough matches (<100)" << std::endl;
        return false;
    }

    // 2. Compute H and F in parallel (simplified to sequential here)
    std::vector<bool> inliers_H, inliers_F;
    float score_H = 0, score_F = 0;
    cv::Mat H21, F21;

    bool H_found = findHomography(inliers_H, score_H, H21);
    bool F_found = findFundamental(inliers_F, score_F, F21);

    std::cout << "Initializer: H score: " << score_H << ", F score: " << score_F << std::endl;

    // 3. Select model
    float ratio = score_H / (score_H + score_F + 1e-5);
    bool use_H = ratio > 0.40; // Threshold from ORB-SLAM

    cv::Mat K = frame_ref_->camera_->K();
    
    bool success = false;
    if (use_H && H_found) {
        std::cout << "Initializer: Selecting Homography" << std::endl;
        success = reconstructH(inliers_H, H21, K, T_c1_c2_, triangulated_points_, is_triangulated_, 1.0, 50);
    } else if (F_found) {
        std::cout << "Initializer: Selecting Fundamental Matrix" << std::endl;
        success = reconstructF(inliers_F, F21, K, T_c1_c2_, triangulated_points_, is_triangulated_, 1.0, 50);
    }

    if (!success) {
        std::cout << "Initializer: Reconstruction failed." << std::endl;
        return false;
    }
    
    // T_c1_c2_ is actually T_c2_c1 (Pose of 2 w.r.t 1)? 
    // Usually reconstruct returns R, t such that x2 = R*x1 + t. This is T_c2_c1.
    // Our convention in Frame is T_cw (World to Camera).
    // If Frame 1 is World (Reference), then T_cw for Frame 2 is T_c2_c1.
    // So T_c1_c2_ member should store T_c2_c1.
    
    return true;
}

bool Initializer::findHomography(std::vector<bool>& inliers, float& score, cv::Mat& H21) {
    if (matches_.size() < 4) return false;
    std::vector<unsigned char> status;
    H21 = cv::findHomography(kps_ref_, kps_cur_, cv::RANSAC, 3.0, status);
    
    score = 0;
    inliers.resize(matches_.size());
    for(size_t i=0; i<status.size(); ++i) {
        if(status[i]) {
            inliers[i] = true;
            score += 1.0; // Simplified score
        } else {
            inliers[i] = false;
        }
    }
    return !H21.empty() && score > 30;
}

bool Initializer::findFundamental(std::vector<bool>& inliers, float& score, cv::Mat& F21) {
    if (matches_.size() < 8) return false;
    std::vector<unsigned char> status;
    F21 = cv::findFundamentalMat(kps_ref_, kps_cur_, cv::FM_RANSAC, 3.0, 0.99, status);

    score = 0;
    inliers.resize(matches_.size());
    for(size_t i=0; i<status.size(); ++i) {
        if(status[i]) {
            inliers[i] = true;
            score += 1.0; // Simplified score
        } else {
            inliers[i] = false;
        }
    }
    return !F21.empty() && score > 30;
}

bool Initializer::reconstructF(std::vector<bool>& inliers, cv::Mat& F21, cv::Mat& K,
                  SE3& T_c2_c1, std::vector<cv::Point3f>& points, std::vector<bool>& triangulated, float min_parallax, int min_triangulated) {
    
    cv::Mat E21 = K.t() * F21 * K;
    cv::Mat R, t;
    
    // Recover pose
    // Note: This returns valid R and t only if E is valid.
    // There are 4 solutions, we need to check chirality.
    // Using OpenCV's recoverPose
    
    std::vector<cv::Point2f> inlier_kps_ref, inlier_kps_cur;
    for(size_t i=0; i<inliers.size(); ++i) {
        if(inliers[i]) {
            inlier_kps_ref.push_back(kps_ref_[i]);
            inlier_kps_cur.push_back(kps_cur_[i]);
        }
    }
    
    if (inlier_kps_ref.size() < (size_t)min_triangulated) return false;

    cv::Mat mask;
    double focal = K.at<double>(0,0);
    cv::Point2d pp(K.at<double>(0,2), K.at<double>(1,2));
    
    int valid_points = cv::recoverPose(E21, inlier_kps_ref, inlier_kps_cur, R, t, focal, pp, mask);
    
    std::cout << "RecoverPose valid points: " << valid_points << std::endl;

    if (valid_points < min_triangulated) return false;

    // Convert to Sophus SE3
    Eigen::Matrix3d R_eig;
    Eigen::Vector3d t_eig;
    cv::cv2eigen(R, R_eig);
    cv::cv2eigen(t, t_eig);
    
    // Normalize rotation to satisfy Sophus orthogonality check
    Eigen::Quaterniond q(R_eig);
    q.normalize();
    
    T_c2_c1 = SE3(q, t_eig); // T_c2_c1 transforms point from 1 to 2.

    // Triangulate
    // For simplicity, re-triangulate all inliers of recoverPose
    points.resize(matches_.size());
    triangulated.assign(matches_.size(), false);
    is_triangulated_ = triangulated;
    
    // Get Projection Matrices
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // P1 = [I|0]
    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    
    // Apply K
    P1 = K * P1;
    P2 = K * P2;
    
    // Mapping from inlier index to original index is needed if we use recoverPose's mask
    // But recoverPose takes compact vectors.
    // Let's iterate original matches and check inliers AND recoverPose mask
    
    int compact_idx = 0;
    for(size_t i=0; i<matches_.size(); ++i) {
        if(!inliers[i]) continue;
        
        if (mask.at<unsigned char>(compact_idx)) {
             // Triangulate single point
             cv::Mat pts4D;
             std::vector<cv::Point2f> pt1 = {kps_ref_[i]};
             std::vector<cv::Point2f> pt2 = {kps_cur_[i]};
             cv::triangulatePoints(P1, P2, pt1, pt2, pts4D);
             
             // Convert to 3D
             pts4D /= pts4D.at<double>(3,0);
             cv::Point3f p3d(pts4D.at<double>(0,0), pts4D.at<double>(1,0), pts4D.at<double>(2,0));
             
             // Check Parallax and Depth positive
             if (p3d.z > 0) {
                 triangulated_points_.push_back(p3d);
                 is_triangulated_[i] = true;
             }
        }
        compact_idx++;
    }

    return true;
}

bool Initializer::reconstructH(std::vector<bool>& inliers, cv::Mat& H21, cv::Mat& K,
                  SE3& T_c2_c1, std::vector<cv::Point3f>& points, std::vector<bool>& triangulated, float min_parallax, int min_triangulated) {
    
    std::vector<cv::Mat> Rs, ts, normals;
    int solutions = cv::decomposeHomographyMat(H21, K, Rs, ts, normals);
    
    cv::Mat T1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1 = K * T1;
    
    double best_good_points = 0;
    int best_solution = -1;
    std::vector<bool> best_triangulated;
    std::vector<cv::Point3f> best_points;
    
    for (int i=0; i<solutions; ++i) {
        cv::Mat R = Rs[i];
        cv::Mat t = ts[i];
        
        cv::Mat T2 = cv::Mat::eye(3, 4, CV_64F);
        R.copyTo(T2.rowRange(0,3).colRange(0,3));
        t.copyTo(T2.rowRange(0,3).col(3));
        cv::Mat P2 = K * T2;
        
        std::vector<bool> current_triangulated(matches_.size(), false);
        std::vector<cv::Point3f> current_points(matches_.size());
        int n_good = 0;
        
        for(size_t j=0; j<matches_.size(); ++j) {
            if(!inliers[j]) continue;
            
            std::vector<cv::Point2f> pt1 = {kps_ref_[j]};
            std::vector<cv::Point2f> pt2 = {kps_cur_[j]};
            cv::Mat pts4D;
            cv::triangulatePoints(P1, P2, pt1, pt2, pts4D);
            
            // Convert to 3D and check depth
            if(pts4D.at<double>(3,0) == 0) continue;
            cv::Point3f p3d(pts4D.at<double>(0,0)/pts4D.at<double>(3,0),
                            pts4D.at<double>(1,0)/pts4D.at<double>(3,0),
                            pts4D.at<double>(2,0)/pts4D.at<double>(3,0));
                            
            if (p3d.z > 0) {
                // Check depth in second camera
                cv::Mat p3d_mat(3, 1, CV_64F);
                p3d_mat.at<double>(0,0) = p3d.x;
                p3d_mat.at<double>(1,0) = p3d.y;
                p3d_mat.at<double>(2,0) = p3d.z;
                
                cv::Mat p3d_c2 = R * p3d_mat + t;
                if (p3d_c2.at<double>(2,0) > 0) {
                    current_triangulated[j] = true;
                    current_points[j] = p3d;
                    n_good++;
                }
            }
        }
        
        if (n_good > best_good_points && n_good > min_triangulated) {
            best_good_points = n_good;
            best_solution = i;
            best_triangulated = current_triangulated;
            best_points = current_points;
        }
    }
    
    if (best_solution != -1) {
        // Output result
        Eigen::Matrix3d R_eig;
        Eigen::Vector3d t_eig;
        cv::cv2eigen(Rs[best_solution], R_eig);
        cv::cv2eigen(ts[best_solution], t_eig);
        
        // Normalize rotation to satisfy Sophus orthogonality check
        Eigen::Quaterniond q(R_eig);
        q.normalize();
        
        // Explicitly construct SO3
        Sophus::SO3d so3(q);
        T_c2_c1 = SE3(so3, t_eig);
        
        triangulated = best_triangulated;
        points = best_points;
        is_triangulated_ = triangulated;
        triangulated_points_ = points;
        
        return true;
    }

    return false;
}

}
