#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/camera.h"
#include "core/frame.h"

using namespace svslam;

namespace {

Camera::Ptr makeCamera() {
    return std::make_shared<Camera>(517.3, 516.5, 318.6, 255.3);
}

cv::Mat makeFeatureImage() {
    cv::Mat image = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::rectangle(image, cv::Rect(30, 30, 80, 60), cv::Scalar(255), cv::FILLED);
    cv::circle(image, cv::Point(220, 80), 24, cv::Scalar(180), cv::FILLED);
    cv::line(image, cv::Point(40, 200), cv::Point(280, 180), cv::Scalar(255), 4);
    cv::putText(image, "SV", cv::Point(120, 150), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                cv::Scalar(220), 2);
    return image;
}

}  // namespace

TEST(FrameTest, ExtractOrbIsDeterministicForSameInput) {
    const auto camera = makeCamera();
    const cv::Mat image = makeFeatureImage();
    const auto detector = cv::ORB::create(300);

    Frame first(0, 0.0, camera, image);
    Frame second(1, 0.0, camera, image);

    first.extractORB(detector);
    second.extractORB(detector);

    ASSERT_EQ(first.keypoints_.size(), second.keypoints_.size());
    ASSERT_EQ(first.descriptors_.rows, second.descriptors_.rows);
    ASSERT_EQ(first.descriptors_.cols, second.descriptors_.cols);

    for (std::size_t i = 0; i < first.keypoints_.size(); ++i) {
        EXPECT_FLOAT_EQ(first.keypoints_[i].pt.x, second.keypoints_[i].pt.x);
        EXPECT_FLOAT_EQ(first.keypoints_[i].pt.y, second.keypoints_[i].pt.y);
        EXPECT_EQ(first.keypoints_[i].octave, second.keypoints_[i].octave);
    }
    EXPECT_EQ(cv::countNonZero(first.descriptors_ != second.descriptors_), 0);
}

TEST(FrameTest, GetDepthSupportsTumUint16Images) {
    Frame frame(0, 0.0, makeCamera(), cv::Mat::zeros(4, 4, CV_8UC1));
    frame.depth_image_ = cv::Mat::zeros(4, 4, CV_16UC1);
    frame.depth_image_.at<std::uint16_t>(2, 1) = 7500;

    EXPECT_FLOAT_EQ(frame.getDepth(1.1f, 1.9f), 1.5f);
    EXPECT_FLOAT_EQ(frame.getDepth(3.0f, 3.0f), -1.0f);
}

TEST(FrameTest, GetDepthSupportsFloatDepthImages) {
    Frame frame(0, 0.0, makeCamera(), cv::Mat::zeros(4, 4, CV_8UC1));
    frame.depth_image_ = cv::Mat::zeros(4, 4, CV_32FC1);
    frame.depth_image_.at<float>(1, 2) = 2.25f;

    EXPECT_FLOAT_EQ(frame.getDepth(1.6f, 0.8f), 2.25f);
    frame.depth_image_.at<float>(1, 2) = std::numeric_limits<float>::infinity();
    EXPECT_FLOAT_EQ(frame.getDepth(1.6f, 0.8f), -1.0f);
}

TEST(FrameTest, BackprojectWithDepthRoundTripsKnownPose) {
    const auto camera = makeCamera();
    Frame frame(0, 0.0, camera, cv::Mat::zeros(240, 320, CV_8UC1));
    const SE3 pose(Eigen::Quaterniond::Identity(), Vec3(0.2, -0.1, 0.3));
    frame.setPose(pose);

    const Vec3 world_point(0.5, -0.2, 4.5);
    const Vec3 point_camera = pose * world_point;
    const Vec2 pixel = camera->project(point_camera);

    const cv::KeyPoint keypoint(
        cv::Point2f(static_cast<float>(pixel.x()), static_cast<float>(pixel.y())), 20.0f);
    const Vec3 reconstructed =
        frame.backprojectWithDepth(keypoint, static_cast<float>(point_camera.z()));

    EXPECT_NEAR(reconstructed.x(), world_point.x(), 1e-5);
    EXPECT_NEAR(reconstructed.y(), world_point.y(), 1e-5);
    EXPECT_NEAR(reconstructed.z(), world_point.z(), 1e-5);
}
