#include <gtest/gtest.h>
#include "backend/optimizer.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include "core/camera.h"
#include "core/frame.h"
#include "core/map.h"
#include <cmath>
#include <random>

using namespace svslam;

static Keyframe::Ptr makeKF(unsigned long id, Camera::Ptr cam, const SE3& pose) {
    auto frame = std::make_shared<Frame>(id, 0.0, cam, cv::Mat::zeros(480, 640, CV_8UC1));
    frame->T_cw_ = pose;
    frame->keypoints_.resize(10);
    frame->landmarks_.resize(10);
    auto kf = std::make_shared<Keyframe>(frame);
    return kf;
}

TEST(OptimizerTest, BundleAdjustmentMovesLandmarksCloserToGroundTruth) {
    // Camera with TUM freiburg1-like parameters
    auto cam = std::make_shared<Camera>(517.3, 516.5, 318.6, 255.3);

    // Ground truth landmark positions (in front of both cameras, z > 0)
    std::vector<Vec3> gt_points = {
        {0.5, 0.3, 3.0},
        {-0.4, 0.2, 2.5},
        {0.1, -0.5, 4.0},
        {0.8, 0.1, 3.5},
        {-0.2, -0.3, 2.0},
    };

    // KF0: identity pose (world origin)
    SE3 pose0;  // identity
    auto kf0 = makeKF(0, cam, pose0);

    // KF1: translated 0.3m to the right
    SE3 pose1(Eigen::Quaterniond::Identity(), Vec3(0.3, 0.0, 0.0));
    auto kf1 = makeKF(1, cam, pose1);

    // Create landmarks with noisy positions
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    std::vector<Landmark::Ptr> landmarks;
    for (size_t i = 0; i < gt_points.size(); ++i) {
        Vec3 noisy_pos = gt_points[i] + Vec3(noise(rng), noise(rng), noise(rng));
        // Ensure z stays positive
        noisy_pos.z() = std::max(noisy_pos.z(), 0.5);
        auto lm = std::make_shared<Landmark>(static_cast<unsigned long>(i), noisy_pos);

        // Project ground truth point into each KF to get observations
        Vec3 p_c0 = pose0 * gt_points[i];
        Vec3 p_c1 = pose1 * gt_points[i];

        Vec2 px0 = cam->project(p_c0);
        Vec2 px1 = cam->project(p_c1);

        // Set keypoint positions for the observations
        kf0->keypoints_[i].pt = cv::Point2f(static_cast<float>(px0.x()), static_cast<float>(px0.y()));
        kf1->keypoints_[i].pt = cv::Point2f(static_cast<float>(px1.x()), static_cast<float>(px1.y()));

        // Add observations
        lm->addObservation(kf0, i);
        lm->addObservation(kf1, i);

        landmarks.push_back(lm);
    }

    // Compute initial total error
    double initial_error = 0.0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        initial_error += (landmarks[i]->getPos() - gt_points[i]).squaredNorm();
    }

    // Run BA
    std::vector<Keyframe::Ptr> keyframes = {kf0, kf1};
    Optimizer::bundleAdjustment(keyframes, landmarks, 20);

    // Compute final total error
    double final_error = 0.0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        final_error += (landmarks[i]->getPos() - gt_points[i]).squaredNorm();
    }

    // BA should reduce the error
    EXPECT_LT(final_error, initial_error)
        << "BA should move landmarks closer to ground truth. "
        << "Initial error: " << initial_error << ", Final error: " << final_error;
}

TEST(OptimizerTest, PoseGraphOptimizationAppliesKnownRelativePose) {
    auto cam = std::make_shared<Camera>(517.3, 516.5, 318.6, 255.3);
    auto map = std::make_shared<Map>();

    auto kf0 = makeKF(0, cam, SE3());
    auto kf1 = makeKF(1, cam, SE3(Eigen::Quaterniond::Identity(), Vec3(1.5, 0.0, 0.0)));
    map->addKeyframe(kf0);
    map->addKeyframe(kf1);

    Optimizer::PoseGraphEdge loop_edge;
    loop_edge.from = kf0;
    loop_edge.to = kf1;
    loop_edge.relative_pose = Sim3(1.0, Eigen::Quaterniond::Identity(), Vec3(1.0, 0.0, 0.0));
    loop_edge.translation_weight = 10.0;
    loop_edge.rotation_weight = 10.0;
    loop_edge.scale_weight = 15.0;

    Optimizer::poseGraphOptimization(map, {loop_edge}, 40, false);

    EXPECT_NEAR(kf1->T_cw_.translation().x(), 1.0, 1e-3);
    EXPECT_NEAR(kf1->T_cw_.translation().y(), 0.0, 1e-6);
    EXPECT_NEAR(kf1->T_cw_.translation().z(), 0.0, 1e-6);
}

TEST(OptimizerTest, DepthPriorCostMatchesObservedDepth) {
    const double camera_pose[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double point[3] = {0.0, 0.0, 2.0};
    double residuals[1] = {0.0};

    DepthPriorError error(2.0, 517.3, 516.5, 318.6, 255.3, 320.0, 240.0, 50.0);
    ASSERT_TRUE(error(camera_pose, point, residuals));
    EXPECT_DOUBLE_EQ(residuals[0], 0.0);

    const double shifted_point[3] = {0.0, 0.0, 2.2};
    ASSERT_TRUE(error(camera_pose, shifted_point, residuals));
    EXPECT_NEAR(residuals[0], 10.0, 1e-9);
}

TEST(OptimizerTest, GravityPriorCostPenalizesTiltError) {
    const double identity_pose[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double residuals[3] = {0.0, 0.0, 0.0};

    GravityPriorError aligned_gravity(0.0, 0.0, -1.0, 5.0);
    ASSERT_TRUE(aligned_gravity(identity_pose, residuals));
    EXPECT_DOUBLE_EQ(residuals[0], 0.0);
    EXPECT_DOUBLE_EQ(residuals[1], 0.0);
    EXPECT_DOUBLE_EQ(residuals[2], 0.0);

    const Eigen::Quaterniond tilt_quaternion(
        Eigen::AngleAxisd(std::acos(-1.0) / 2.0, Eigen::Vector3d::UnitX()));
    const double tilted_pose[7] = {
        0.0, 0.0, 0.0,
        tilt_quaternion.w(), tilt_quaternion.x(), tilt_quaternion.y(), tilt_quaternion.z()};
    ASSERT_TRUE(aligned_gravity(tilted_pose, residuals));
    EXPECT_GT(std::abs(residuals[0]) + std::abs(residuals[1]) + std::abs(residuals[2]), 1.0);
}
