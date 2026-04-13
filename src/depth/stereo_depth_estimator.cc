#include "depth/stereo_depth_estimator.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace svslam {

StereoDepthEstimator::StereoDepthEstimator(double baseline_meters, const cv::Mat& K)
    : StereoDepthEstimator(
          baseline_meters,
          K.empty() ? 0.0 : K.at<double>(0, 0),
          K.empty() ? 0.0 : K.at<double>(1, 1),
          K.empty() ? 0.0 : K.at<double>(0, 2),
          K.empty() ? 0.0 : K.at<double>(1, 2)) {}

StereoDepthEstimator::StereoDepthEstimator(double baseline_meters,
                                           double fx,
                                           double fy,
                                           double cx,
                                           double cy)
    : baseline_meters_(baseline_meters), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

cv::Mat StereoDepthEstimator::estimate(const cv::Mat& image) {
    (void)image;
    return cv::Mat();
}

cv::Mat StereoDepthEstimator::estimate(const cv::Mat& left_image, const cv::Mat& right_image) {
    if (left_image.empty() || right_image.empty() || left_image.size() != right_image.size()) {
        return cv::Mat();
    }
    if (baseline_meters_ <= 0.0 || fx_ <= 0.0 || left_image.cols < 32 || left_image.rows < 16) {
        return cv::Mat();
    }

    const cv::Mat left_gray = toGrayscale(left_image);
    const cv::Mat right_gray = toGrayscale(right_image);
    if (left_gray.empty() || right_gray.empty()) {
        return cv::Mat();
    }

    cv::Mat disparity_fixed;
    createMatcher(left_gray.cols)->compute(left_gray, right_gray, disparity_fixed);

    cv::Mat disparity;
    disparity_fixed.convertTo(disparity, CV_32FC1, 1.0 / 16.0);

    cv::Mat depth(left_gray.size(), CV_32FC1, cv::Scalar(0.0f));
    const float depth_scale = static_cast<float>(baseline_meters_ * fx_);
    for (int y = 0; y < disparity.rows; ++y) {
        const float* disparity_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float disparity_value = disparity_row[x];
            if (!std::isfinite(disparity_value) || disparity_value <= 0.0f) {
                continue;
            }

            const float depth_value = depth_scale / disparity_value;
            if (!std::isfinite(depth_value) || depth_value < kMinDepthMeters || depth_value > kMaxDepthMeters) {
                continue;
            }
            depth_row[x] = depth_value;
        }
    }

    return depth;
}

cv::Mat StereoDepthEstimator::toGrayscale(const cv::Mat& image) const {
    if (image.channels() == 1) {
        return image;
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        return gray;
    }

    return cv::Mat();
}

int StereoDepthEstimator::resolveNumDisparities(int image_width) const {
    const double desired_max_disparity = (baseline_meters_ * fx_) / kMinDepthMeters;
    const int rounded_desired = static_cast<int>(std::ceil(desired_max_disparity / 16.0)) * 16;
    const int max_supported = std::max(16, ((image_width - 1) / 16) * 16);
    return std::max(16, std::min(rounded_desired, max_supported));
}

cv::Ptr<cv::StereoSGBM> StereoDepthEstimator::createMatcher(int image_width) const {
    const int block_size = 5;
    const int num_disparities = resolveNumDisparities(image_width);

    cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0, num_disparities, block_size);
    matcher->setP1(8 * block_size * block_size);
    matcher->setP2(32 * block_size * block_size);
    matcher->setPreFilterCap(31);
    matcher->setUniquenessRatio(10);
    matcher->setSpeckleWindowSize(50);
    matcher->setSpeckleRange(2);
    matcher->setDisp12MaxDiff(1);
    matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    return matcher;
}

}  // namespace svslam
