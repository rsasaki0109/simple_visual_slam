#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "depth/stereo_depth_estimator.h"

using namespace svslam;

namespace {

cv::Mat make_textured_image(int width, int height) {
    cv::Mat image(height, width, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(image, cv::RNG::UNIFORM, 0, 255);
    return image;
}

cv::Mat shift_left(const cv::Mat& image, int disparity_pixels) {
    cv::Mat shifted(image.size(), image.type(), cv::Scalar(0));
    image.colRange(disparity_pixels, image.cols).copyTo(shifted.colRange(0, image.cols - disparity_pixels));
    return shifted;
}

float median_valid_depth(const cv::Mat& depth, const cv::Rect& roi) {
    std::vector<float> values;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            const float value = depth.at<float>(y, x);
            if (value > 0.0f && std::isfinite(value)) {
                values.push_back(value);
            }
        }
    }

    if (values.empty()) {
        return 0.0f;
    }

    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

}  // namespace

TEST(StereoDepthEstimatorTest, ComputesMetricDepthFromKnownDisparity) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 96;
    constexpr int kDisparityPixels = 8;
    constexpr double kBaselineMeters = 0.1;
    constexpr double kFx = 60.0;
    const float expected_depth_meters =
        static_cast<float>((kBaselineMeters * kFx) / kDisparityPixels);

    const cv::Mat left = make_textured_image(kWidth, kHeight);
    const cv::Mat right = shift_left(left, kDisparityPixels);

    StereoDepthEstimator estimator(kBaselineMeters, kFx, kFx, kWidth / 2.0, kHeight / 2.0);
    const cv::Mat depth = estimator.estimate(left, right);

    ASSERT_FALSE(depth.empty());
    ASSERT_EQ(depth.type(), CV_32FC1);

    const cv::Rect center_roi(32, 16, 96, 64);
    const float median_depth = median_valid_depth(depth, center_roi);
    ASSERT_GT(median_depth, 0.0f);
    EXPECT_NEAR(median_depth, expected_depth_meters, 0.15f);
}

TEST(StereoDepthEstimatorTest, FiltersInvalidDisparitiesToZeroDepth) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 96;
    constexpr double kBaselineMeters = 0.1;
    constexpr double kFx = 60.0;

    const cv::Mat left = make_textured_image(kWidth, kHeight);
    const cv::Mat right(kHeight, kWidth, CV_8UC1, cv::Scalar(0));

    StereoDepthEstimator estimator(kBaselineMeters, kFx, kFx, kWidth / 2.0, kHeight / 2.0);
    const cv::Mat depth = estimator.estimate(left, right);

    ASSERT_FALSE(depth.empty());
    EXPECT_EQ(cv::countNonZero(depth > 0.0f), 0);
}
