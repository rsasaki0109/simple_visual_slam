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

TEST(OptimizerTest, VelocityPreintegrationResidualZeroWhenPredictionMatches) {
    // Zero-gravity setup: KF_i at world origin with v=0; KF_j at world (1,0,0)
    // with v=(2,0,0). Preintegration says delta_p=(1,0,0), delta_v=(2,0,0)
    // over dt=1 s. With accel bias matching the reference, the residual
    // must vanish because the prediction is exact.
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double pose_j[7] = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double vel_i[3] = {0.0, 0.0, 0.0};
    const double vel_j[3] = {2.0, 0.0, 0.0};
    const double ba_i[3] = {0.0, 0.0, 0.0};  // equal to ba_ref → no correction
    double residuals[9] = {0.0};

    VelocityPreintegrationError error(
        Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0),
        Eigen::Quaterniond::Identity(),
        Vec3::Zero(), 1.0, Vec3::Zero(),
        Eigen::Quaterniond::Identity(), Vec3::Zero(),
        /*pos_weight=*/10.0, /*vel_weight=*/5.0, /*rot_weight=*/4.0);
    ASSERT_TRUE(error(pose_i, pose_j, vel_i, vel_j, ba_i, residuals));
    for (int k = 0; k < 9; ++k) {
        EXPECT_NEAR(residuals[k], 0.0, 1e-9) << "residual[" << k << "]";
    }
}

TEST(OptimizerTest, VelocityPreintegrationResidualAccountsForGravity) {
    // Free-fall: v=0 for both KFs, zero body-frame preintegration deltas,
    // KF_j still at origin after 1 s → the residual should highlight the
    // missing gravity drop.
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double pose_j[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double vel_i[3] = {0.0, 0.0, 0.0};
    const double vel_j[3] = {0.0, 0.0, 0.0};
    const double ba_i[3] = {0.0, 0.0, 0.0};
    double residuals[9] = {0.0};

    VelocityPreintegrationError error(
        Vec3::Zero(), Vec3::Zero(),
        Eigen::Quaterniond::Identity(),
        Vec3::Zero(), 1.0,
        Vec3(0.0, 0.0, -9.81), Eigen::Quaterniond::Identity(), Vec3::Zero(),
        /*pos_weight=*/1.0, /*vel_weight=*/1.0, /*rot_weight=*/1.0);
    ASSERT_TRUE(error(pose_i, pose_j, vel_i, vel_j, ba_i, residuals));
    EXPECT_NEAR(residuals[2], 4.905, 1e-6);
    EXPECT_NEAR(residuals[5], 9.81, 1e-6);
    // No rotation motion, identity delta_R → rotation residuals are zero.
    EXPECT_NEAR(residuals[6], 0.0, 1e-9);
    EXPECT_NEAR(residuals[7], 0.0, 1e-9);
    EXPECT_NEAR(residuals[8], 0.0, 1e-9);
}

TEST(OptimizerTest, VelocityPreintegrationBiasAppliesFirstOrderCorrection) {
    // Same stationary-poses setup as the gravity test but now the accel bias
    // at KF_i differs from the reference bias used during preintegration. The
    // first-order correction should show up as -(bias-ref)*dt in the velocity
    // residual and -0.5*(bias-ref)*dt^2 in the position residual.
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double pose_j[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double vel_i[3] = {0.0, 0.0, 0.0};
    const double vel_j[3] = {0.0, 0.0, 0.0};
    const double ba_i[3] = {0.1, 0.0, 0.0};
    double residuals[9] = {0.0};

    // Pre-integration stored deltas assuming bias = 0; no gravity.
    VelocityPreintegrationError error(
        Vec3::Zero(), Vec3::Zero(),
        Eigen::Quaterniond::Identity(),
        Vec3::Zero(), /*dt=*/2.0,
        Vec3::Zero(), Eigen::Quaterniond::Identity(), Vec3::Zero(),
        /*pos_weight=*/1.0, /*vel_weight=*/1.0, /*rot_weight=*/1.0);
    ASSERT_TRUE(error(pose_i, pose_j, vel_i, vel_j, ba_i, residuals));
    // With positions fixed at origin the residual is -(−R*dp_corr) = R*dp_corr.
    // R=I here, dp_corr = dp - 0.5*dba*dt^2 = -0.5 * 0.1 * 4 = -0.2 on x.
    // residual_pos_x = (p_j-p_i) - 0 - 0 - R*dp_corr = 0 - (-0.2) = 0.2.
    EXPECT_NEAR(residuals[0], 0.2, 1e-9);
    // residual_vel_x = (v_j-v_i) - 0 - R*dv_corr. dv_corr = 0 - 0.1*2 = -0.2.
    // residual_vel_x = 0 - (-0.2) = 0.2.
    EXPECT_NEAR(residuals[3], 0.2, 1e-9);
}

TEST(OptimizerTest, VelocityPreintegrationResidualZeroWhenRotationMatches) {
    // Both KFs at the world origin. KF_j is rotated 30° about Z vs KF_i. The
    // preintegrated delta_R encodes the same 30° rotation, so the rotation
    // residual should be zero. Position + velocity residuals are set up to
    // match as well (no motion, no gravity, zero bias).
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const Eigen::Quaterniond q_wb_j(
        Eigen::AngleAxisd(30.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
    // pose_j stores T_cw_j, and we want R_wb_j = q_wb_j with T_cam_imu = I
    // (so q_wb == q_wc). Thus q_cw_j = q_wc_j^T = q_wb_j^T.
    const Eigen::Quaterniond q_cw_j = q_wb_j.conjugate();
    const double pose_j[7] = {0.0, 0.0, 0.0,
                              q_cw_j.w(), q_cw_j.x(), q_cw_j.y(), q_cw_j.z()};
    const double vel_i[3] = {0.0, 0.0, 0.0};
    const double vel_j[3] = {0.0, 0.0, 0.0};
    const double ba_i[3] = {0.0, 0.0, 0.0};
    double residuals[9] = {0.0};

    VelocityPreintegrationError error(
        Vec3::Zero(), Vec3::Zero(),
        q_wb_j,  // delta_R matches the rotation from i to j
        Vec3::Zero(), /*dt=*/1.0, Vec3::Zero(),
        Eigen::Quaterniond::Identity(), Vec3::Zero(),
        /*pos_weight=*/1.0, /*vel_weight=*/1.0, /*rot_weight=*/10.0);
    ASSERT_TRUE(error(pose_i, pose_j, vel_i, vel_j, ba_i, residuals));
    for (int k = 0; k < 9; ++k) {
        EXPECT_NEAR(residuals[k], 0.0, 1e-9) << "residual[" << k << "]";
    }
}

TEST(OptimizerTest, VelocityPreintegrationResidualPenalizesRotationMismatch) {
    // Predicted 30° rotation but actual is 40° — residual should fire with
    // magnitude ≈ weight * 10° in radians = weight * 0.1745.
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const Eigen::Quaterniond q_actual(
        Eigen::AngleAxisd(40.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_predicted(
        Eigen::AngleAxisd(30.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_cw_j = q_actual.conjugate();
    const double pose_j[7] = {0.0, 0.0, 0.0,
                              q_cw_j.w(), q_cw_j.x(), q_cw_j.y(), q_cw_j.z()};
    const double vel_i[3] = {0.0, 0.0, 0.0};
    const double vel_j[3] = {0.0, 0.0, 0.0};
    const double ba_i[3] = {0.0, 0.0, 0.0};
    double residuals[9] = {0.0};

    const double rot_weight = 5.0;
    VelocityPreintegrationError error(
        Vec3::Zero(), Vec3::Zero(),
        q_predicted,
        Vec3::Zero(), /*dt=*/1.0, Vec3::Zero(),
        Eigen::Quaterniond::Identity(), Vec3::Zero(),
        /*pos_weight=*/1.0, /*vel_weight=*/1.0, rot_weight);
    ASSERT_TRUE(error(pose_i, pose_j, vel_i, vel_j, ba_i, residuals));
    // pos and vel components are unaffected by the rotation mismatch here.
    for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(residuals[k], 0.0, 1e-6);
    }
    // The error rotation is 10° about Z; its log (half-angle sin) gives
    // 2 * sin(5°) ≈ 0.1745 on the z residual (x, y stay 0).
    EXPECT_NEAR(residuals[6], 0.0, 1e-6);
    EXPECT_NEAR(residuals[7], 0.0, 1e-6);
    EXPECT_NEAR(residuals[8],
                rot_weight * 2.0 * std::sin(5.0 * M_PI / 180.0), 1e-6);
}

TEST(OptimizerTest, BiasAnchorErrorPenalizesNonZeroBias) {
    const double bias[3] = {0.1, -0.2, 0.3};
    double residuals[3] = {0.0, 0.0, 0.0};

    BiasAnchorError error(/*weight=*/2.0);
    ASSERT_TRUE(error(bias, residuals));
    EXPECT_NEAR(residuals[0], 0.2, 1e-12);
    EXPECT_NEAR(residuals[1], -0.4, 1e-12);
    EXPECT_NEAR(residuals[2], 0.6, 1e-12);
}

TEST(OptimizerTest, BiasRandomWalkErrorMeasuresDifference) {
    const double bias_i[3] = {0.1, 0.0, 0.0};
    const double bias_j[3] = {0.15, 0.0, 0.0};
    double residuals[3] = {0.0, 0.0, 0.0};

    BiasRandomWalkError error(/*weight=*/10.0);
    ASSERT_TRUE(error(bias_i, bias_j, residuals));
    EXPECT_NEAR(residuals[0], 0.5, 1e-9);
    EXPECT_NEAR(residuals[1], 0.0, 1e-12);
    EXPECT_NEAR(residuals[2], 0.0, 1e-12);
}

TEST(OptimizerTest, VelocityDeltaPriorZeroWhenPositionsMatchVelocityIntegral) {
    // KF_i at world origin, KF_j at world (1, 0, 0), velocity 0.5 m/s over 2 s.
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double pose_j[7] = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double residuals[3] = {0.0, 0.0, 0.0};

    VelocityDeltaPriorError error(Vec3(0.5, 0.0, 0.0), 2.0, /*weight=*/10.0);
    ASSERT_TRUE(error(pose_i, pose_j, residuals));
    EXPECT_NEAR(residuals[0], 0.0, 1e-9);
    EXPECT_NEAR(residuals[1], 0.0, 1e-9);
    EXPECT_NEAR(residuals[2], 0.0, 1e-9);
}

TEST(OptimizerTest, VelocityDeltaPriorPenalizesJumpBeyondExpectedDelta) {
    // Same pair but velocity says we should have moved 2.0 m while we only
    // moved 1.0 m — expect a 1.0 m residual on x scaled by weight.
    const double pose_i[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    const double pose_j[7] = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double residuals[3] = {0.0, 0.0, 0.0};

    VelocityDeltaPriorError error(Vec3(1.0, 0.0, 0.0), 2.0, /*weight=*/5.0);
    ASSERT_TRUE(error(pose_i, pose_j, residuals));
    EXPECT_NEAR(residuals[0], 5.0 * (1.0 - 2.0), 1e-9);
    EXPECT_NEAR(residuals[1], 0.0, 1e-9);
    EXPECT_NEAR(residuals[2], 0.0, 1e-9);
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
