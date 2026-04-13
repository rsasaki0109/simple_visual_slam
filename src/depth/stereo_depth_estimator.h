#pragma once

#include <opencv2/calib3d.hpp>

#include "depth/depth_estimator.h"

namespace svslam {

class StereoDepthEstimator : public DepthEstimator {
public:
    StereoDepthEstimator(double baseline_meters, const cv::Mat& K);
    StereoDepthEstimator(double baseline_meters, double fx, double fy, double cx, double cy);
    ~StereoDepthEstimator() override = default;

    cv::Mat estimate(const cv::Mat& image) override;
    cv::Mat estimate(const cv::Mat& left_image, const cv::Mat& right_image) override;
    bool isMetric() const override { return true; }

private:
    static constexpr float kMinDepthMeters = 0.1f;
    static constexpr float kMaxDepthMeters = 20.0f;

    cv::Mat toGrayscale(const cv::Mat& image) const;
    int resolveNumDisparities(int image_width) const;
    cv::Ptr<cv::StereoSGBM> createMatcher(int image_width) const;

    double baseline_meters_ = 0.0;
    double fx_ = 0.0;
    double fy_ = 0.0;
    double cx_ = 0.0;
    double cy_ = 0.0;
};

}  // namespace svslam
