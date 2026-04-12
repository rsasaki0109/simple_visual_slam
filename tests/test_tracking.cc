#include <limits>

#include <gtest/gtest.h>

#include "core/map.h"
#include "test_synthetic_scene.h"
#include "tracking/tracking.h"

using namespace svslam;

namespace {

Frame::Ptr makeEmptyFrame(unsigned long id, double timestamp, const Camera::Ptr& camera) {
    auto frame = std::make_shared<Frame>(
        id, timestamp, camera,
        cv::Mat::zeros(test_support::kImageHeight, test_support::kImageWidth, CV_8UC1));
    frame->descriptors_ = cv::Mat(0, 32, CV_8U);
    frame->landmarks_.clear();
    return frame;
}

}  // namespace

TEST(TrackingPolicyTest, RejectsNonFiniteReprojectionInputs) {
    EXPECT_FALSE(Tracking::shouldAcceptRecomputedPose(
        std::numeric_limits<double>::quiet_NaN(), 1.0));
    EXPECT_FALSE(Tracking::shouldAcceptRecomputedPose(
        1.0, std::numeric_limits<double>::infinity()));
}

TEST(TrackingPolicyTest, AcceptsRecoveryUpdateAtSupportRegressionBoundary) {
    EXPECT_TRUE(Tracking::shouldAcceptLocalMapPoseUpdate(
        120, 160, false, 0.18, 0.18, 1));
    EXPECT_FALSE(Tracking::shouldAcceptLocalMapPoseUpdate(
        120, 160, false, 0.19, 0.10, 1));
}

TEST(TrackingStateTest, DepthInitializationTransitionsDirectlyToOk) {
    Tracking tracking;
    tracking.setMap(std::make_shared<Map>());

    const auto camera = test_support::makeTestCamera();
    const auto world_points = test_support::makeSyntheticWorldPoints();
    const auto depth_frame =
        test_support::makeProjectedFrame(0, 0.0, camera, SE3(), world_points, true);

    EXPECT_EQ(tracking.state_, TrackingState::NO_IMAGES_YET);
    EXPECT_TRUE(tracking.addFrame(depth_frame));
    EXPECT_EQ(tracking.state_, TrackingState::OK);
}

TEST(TrackingStateTest, TransitionsFromInitializationToLostAndBackToOk) {
    Tracking tracking;
    tracking.setMap(std::make_shared<Map>());

    const auto camera = test_support::makeTestCamera();
    const auto world_points = test_support::makeSyntheticWorldPoints();

    const auto first_frame =
        test_support::makeProjectedFrame(0, 0.0, camera, SE3(), world_points);
    const SE3 second_pose(Eigen::Quaterniond::Identity(), Vec3(0.30, 0.0, 0.0));
    const auto second_frame =
        test_support::makeProjectedFrame(1, 0.1, camera, second_pose, world_points);

    EXPECT_EQ(tracking.state_, TrackingState::NO_IMAGES_YET);
    EXPECT_TRUE(tracking.addFrame(first_frame));
    EXPECT_EQ(tracking.state_, TrackingState::NOT_INITIALIZED);

    EXPECT_TRUE(tracking.addFrame(second_frame));
    EXPECT_EQ(tracking.state_, TrackingState::OK);

    const auto empty_frame = makeEmptyFrame(2, 0.2, camera);
    EXPECT_FALSE(tracking.addFrame(empty_frame));
    EXPECT_EQ(tracking.state_, TrackingState::LOST);

    // Keep a valid motion-model reference so the recovery frame can exercise LOST -> OK
    // without depending on empty-descriptor matcher behavior.
    tracking.last_frame_ = second_frame;

    const auto recovery_frame =
        test_support::makeProjectedFrame(3, 0.3, camera, second_pose, world_points);
    EXPECT_TRUE(tracking.addFrame(recovery_frame));
    EXPECT_EQ(tracking.state_, TrackingState::OK);
}
