#pragma once

#include "core/common.h"
#include "core/frame.h"
#include <opencv2/opencv.hpp>

namespace svslam {

class Initializer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Initializer>;

    Initializer(Frame::Ptr frame_ref);

    bool initialize(Frame::Ptr frame_cur, float sigma = 1.0, int max_iterations = 200);

    // Results
    SE3 T_c1_c2_; // Pose of frame 2 w.r.t frame 1 (or current w.r.t ref)
    std::vector<cv::Point3f> triangulated_points_; // In frame 1 (ref) coordinates
    std::vector<bool> is_triangulated_; // matches indices
    std::vector<cv::DMatch> matches_;

private:
    bool findHomography(std::vector<bool>& inliers, float& score, cv::Mat& H21);
    bool findFundamental(std::vector<bool>& inliers, float& score, cv::Mat& F21);

    bool reconstructH(std::vector<bool>& inliers, cv::Mat& H21, cv::Mat& K,
                      SE3& T_c2_c1, std::vector<cv::Point3f>& points, std::vector<bool>& triangulated, float min_parallax, int min_triangulated);

    bool reconstructF(std::vector<bool>& inliers, cv::Mat& F21, cv::Mat& K,
                      SE3& T_c2_c1, std::vector<cv::Point3f>& points, std::vector<bool>& triangulated, float min_parallax, int min_triangulated);

    void normalizeKeys(const std::vector<cv::KeyPoint>& keys, std::vector<cv::Point2f>& normalized_points, cv::Mat& T);

    Frame::Ptr frame_ref_;
    std::vector<cv::Point2f> kps_ref_;
    std::vector<cv::Point2f> kps_cur_;
};

}
