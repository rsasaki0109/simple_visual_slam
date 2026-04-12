#include <gtest/gtest.h>

#include "test_synthetic_scene.h"
#include "tracking/initializer.h"

using namespace svslam;

TEST(InitializerTest, InitializesFromSyntheticTwoViewGeometry) {
    const auto camera = test_support::makeTestCamera();
    const auto world_points = test_support::makeSyntheticWorldPoints();

    const auto first_frame =
        test_support::makeProjectedFrame(0, 0.0, camera, SE3(), world_points);
    const SE3 second_pose(Eigen::Quaterniond::Identity(), Vec3(0.30, 0.0, 0.0));
    const auto second_frame =
        test_support::makeProjectedFrame(1, 0.1, camera, second_pose, world_points);

    Initializer initializer(first_frame);
    ASSERT_TRUE(initializer.initialize(second_frame));

    const auto triangulated_count = static_cast<int>(std::count(
        initializer.is_triangulated_.begin(),
        initializer.is_triangulated_.end(),
        true));

    EXPECT_GE(triangulated_count, 50);
    EXPECT_TRUE(initializer.T_c1_c2_.translation().allFinite());
    EXPECT_GT(initializer.T_c1_c2_.translation().norm(), 0.1);
}
