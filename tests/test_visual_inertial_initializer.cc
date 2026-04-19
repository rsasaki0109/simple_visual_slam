#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/frame.h"
#include "core/keyframe.h"
#include "core/camera.h"
#include "sensors/imu_preintegration_span.h"
#include "tracking/visual_inertial_initializer.h"

using namespace svslam;

namespace {

// Build a keyframe with a given world pose, timestamp, and IMU span from
// its predecessor (if any). The span is populated with the body-frame
// deltas that make the VI init equations exactly satisfied at the given
// scale / gravity / velocity profile — we invert the Forster equations:
//
//   R_wb_i * delta_p = s*(p_wb_j - p_wb_i) - v_i*dt - 0.5*g*dt^2
//   R_wb_i * delta_v = (v_j - v_i) - g*dt
//   delta_R          = R_wb_i^T * R_wb_j
//
// which gives a noise-free bootstrap scene that the initializer should
// recover to within numerical precision.
struct KfScene {
    std::vector<Keyframe::Ptr> kfs;
    std::vector<Vec3> velocities_world;
    Vec3 gravity_world;
    double scale = 1.0;
    Vec3 gyro_bias = Vec3::Zero();
};

Keyframe::Ptr makeKeyframeForScene(unsigned long id, double timestamp,
                                   const Camera::Ptr& cam,
                                   const SE3& T_cw) {
    // A bare Frame with a few keypoints; the initializer only reads pose
    // and span, so the imagery + landmarks don't matter here.
    auto frame = std::make_shared<Frame>(id, timestamp, cam,
                                         cv::Mat::zeros(480, 640, CV_8UC1));
    frame->keypoints_.resize(4);
    frame->landmarks_.resize(4);
    frame->T_cw_ = T_cw;
    auto kf = std::make_shared<Keyframe>(frame);
    kf->timestamp_ = timestamp;
    return kf;
}

KfScene buildNoiselessEurocLikeScene(std::size_t num_kfs,
                                     double scale_true,
                                     const Vec3& gravity_world,
                                     const SE3& T_cam_imu) {
    KfScene scene;
    scene.scale = scale_true;
    scene.gravity_world = gravity_world;

    auto cam = std::make_shared<Camera>(458.65, 457.30, 367.22, 248.38);

    const double dt = 0.1;
    Vec3 v_world(0.5, 0.2, 0.0);  // nonzero so the system is observable
    const Vec3 accel_world(0.3, -0.1, 0.0);  // constant accel over window

    // Generate body poses R_wb, p_wb in the "true" metric world frame,
    // then build the visual map at 1/scale units (the mono median-depth
    // rescaling multiplies positions by 1/scale_true by construction).
    const Sophus::SO3d R_cb = T_cam_imu.so3();
    const Vec3 t_cb = T_cam_imu.translation();
    const Sophus::SO3d R_bc = R_cb.inverse();
    const Vec3 t_bc = -(R_bc * t_cb);

    std::vector<Sophus::SO3d> R_wb_list;
    std::vector<Vec3> p_wb_list;
    std::vector<Vec3> v_list;

    Sophus::SO3d R_wb = Sophus::SO3d();  // identity
    Vec3 p_wb(0.0, 0.0, 2.0);  // some initial position in metric world
    R_wb_list.push_back(R_wb);
    p_wb_list.push_back(p_wb);
    v_list.push_back(v_world);

    // Small constant angular velocity to exercise the gyro path.
    const Vec3 omega_world(0.0, 0.0, 0.05);

    for (std::size_t i = 1; i < num_kfs; ++i) {
        const Vec3 v_next = v_world + gravity_world * dt + accel_world * dt;
        const Vec3 p_next = p_wb + v_world * dt + 0.5 * (gravity_world + accel_world) * dt * dt;
        const Sophus::SO3d R_next = R_wb * Sophus::SO3d::exp(omega_world * dt);
        R_wb_list.push_back(R_next);
        p_wb_list.push_back(p_next);
        v_list.push_back(v_next);
        R_wb = R_next;
        p_wb = p_next;
        v_world = v_next;
    }

    // The visual map scale is 1/scale_true (mono median-depth rescale).
    for (std::size_t i = 0; i < num_kfs; ++i) {
        const Sophus::SO3d R_wc = R_wb_list[i] * R_cb.inverse();
        const Vec3 p_wc_metric = p_wb_list[i] + R_wb_list[i] * t_bc;
        const Vec3 p_wc_visual = p_wc_metric / scale_true;
        const SE3 T_wc(R_wc, p_wc_visual);
        auto kf = makeKeyframeForScene(static_cast<unsigned long>(i),
                                       static_cast<double>(i) * dt, cam,
                                       T_wc.inverse());
        scene.kfs.push_back(kf);
        scene.velocities_world.push_back(v_list[i]);
    }

    // Build spans from metric deltas (preintegration is invariant under
    // the visual rescaling — it lives in the body frame).
    for (std::size_t i = 1; i < num_kfs; ++i) {
        const Vec3 dp = R_wb_list[i - 1].inverse() *
                        (p_wb_list[i] - p_wb_list[i - 1] -
                         v_list[i - 1] * dt -
                         0.5 * gravity_world * dt * dt);
        const Vec3 dv = R_wb_list[i - 1].inverse() *
                        (v_list[i] - v_list[i - 1] - gravity_world * dt);
        const Sophus::SO3d dR = R_wb_list[i - 1].inverse() * R_wb_list[i];

        auto span = std::make_unique<ImuPreintegrationSpan>();
        span->delta_R = dR;
        span->delta_v = dv;
        span->delta_p = dp;
        span->dt = dt;
        span->bias_accel = Vec3::Zero();
        span->bias_gyro = Vec3::Zero();
        span->from_kf_id = scene.kfs[i - 1]->id_;
        span->T_cam_imu = T_cam_imu;
        span->valid = true;
        scene.kfs[i]->prev_imu_span_ = std::move(span);
    }

    return scene;
}

}  // namespace

TEST(VisualInertialInitializerTest, RecoversScaleAndGravityOnSyntheticScene) {
    const double scale_true = 0.25;  // visual map 4× the metric magnitude
    const Vec3 gravity_world(0.0, 0.0, -9.81);
    // Non-identity T_cam_imu so the body-vs-camera math is exercised.
    const Sophus::SO3d R_cb(Eigen::Quaterniond(
        Eigen::AngleAxisd(0.1, Eigen::Vector3d(0.0, 1.0, 0.0)) *
        Eigen::AngleAxisd(-0.2, Eigen::Vector3d(1.0, 0.0, 0.0))));
    const Vec3 t_cb(0.05, -0.02, 0.01);
    const SE3 T_cam_imu(R_cb, t_cb);

    auto scene = buildNoiselessEurocLikeScene(20, scale_true, gravity_world, T_cam_imu);

    VisualInertialInitializer::Options opts;
    opts.metric_scale = false;
    opts.scale_prior_weight = 0.0;  // pure LSQ recovery
    VisualInertialInitializer vi(opts);

    const auto result = vi.initialize(scene.kfs);
    ASSERT_TRUE(result.converged) << "message: " << result.message
        << " scale=" << result.scale
        << " rot_rms=" << result.rotation_residual_rms
        << " lin_rms=" << result.linear_residual_rms;

    // Scale should recover to ≈ scale_true within LSQ tolerance.
    EXPECT_NEAR(result.scale, scale_true, 1e-3);

    // Gravity should line up with (0,0,-9.81). Note: the initializer
    // always normalises |g| to the expected magnitude, so we check the
    // direction.
    EXPECT_NEAR(result.gravity_w.norm(), 9.81, 1e-3);
    EXPECT_GT(result.gravity_w.normalized().dot(gravity_world.normalized()),
              1.0 - 1e-3);

    // Velocities should closely match the ground truth profile.
    ASSERT_EQ(result.velocities.size(), scene.velocities_world.size());
    for (std::size_t i = 0; i < result.velocities.size(); ++i) {
        EXPECT_LT((result.velocities[i] - scene.velocities_world[i]).norm(), 1e-3)
            << "velocity mismatch at KF " << i;
    }
}

TEST(VisualInertialInitializerTest, RecoversGyroBiasClosedForm) {
    const double scale_true = 1.0;  // irrelevant here; we just need poses
    const Vec3 gravity_world(0.0, 0.0, -9.81);
    const Sophus::SO3d R_cb;  // identity
    const Vec3 t_cb = Vec3::Zero();
    const SE3 T_cam_imu(R_cb, t_cb);

    // Build a baseline scene, then perturb every span's delta_R to
    // simulate a nonzero gyro bias recorded during preintegration. The
    // closed-form stage-1 solve should back out the bias that explains
    // the rotation residual.
    auto scene = buildNoiselessEurocLikeScene(15, scale_true, gravity_world, T_cam_imu);

    const Vec3 true_bg(0.01, -0.02, 0.005);  // rad/s (EuRoC-scale)
    for (std::size_t i = 1; i < scene.kfs.size(); ++i) {
        auto& span = *scene.kfs[i]->prev_imu_span_;
        // Forster linearization: delta_R(b_ref + db) ≈ delta_R(b_ref) * Exp(-dt * db).
        // If we recorded a span at b_ref = 0 but the true gyro bias is
        // true_bg, then the clean delta_R_true satisfies
        //   delta_R_true ≈ delta_R_recorded * Exp(-dt * true_bg),
        // so working backwards delta_R_recorded ≈ delta_R_true * Exp(+dt * true_bg).
        const Sophus::SO3d perturb = Sophus::SO3d::exp(true_bg * span.dt);
        span.delta_R = span.delta_R * perturb;
        span.bias_gyro = Vec3::Zero();  // recorded during preint at zero bias
    }

    VisualInertialInitializer::Options opts;
    // Stage 1 (gyro bias) is independent of the linear scale solve, so we
    // let the test pin scale=1 to keep the test focused on the rotation
    // side; scale recovery is covered by the dedicated scene test.
    opts.metric_scale = true;
    opts.rotation_residual_rms_max = 0.5;  // accept larger residual here
    VisualInertialInitializer vi(opts);

    const auto result = vi.initialize(scene.kfs);
    ASSERT_TRUE(result.converged) << "message: " << result.message;

    // The estimated gyro bias should be close to the perturbation bias.
    EXPECT_LT((result.gyro_bias - true_bg).norm(), 5e-3)
        << "estimated gyro bias = " << result.gyro_bias.transpose()
        << ", expected ≈ " << true_bg.transpose();
}

TEST(VisualInertialInitializerTest, RejectsWindowWithoutSpans) {
    auto cam = std::make_shared<Camera>(458.65, 457.30, 367.22, 248.38);
    std::vector<Keyframe::Ptr> kfs;
    for (unsigned long i = 0; i < 5; ++i) {
        kfs.push_back(makeKeyframeForScene(i, static_cast<double>(i) * 0.1, cam,
                                           SE3()));
    }

    VisualInertialInitializer vi;
    const auto result = vi.initialize(kfs);
    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.message.empty());
}

TEST(VisualInertialInitializerTest, MetricScaleModePinsScaleToOne) {
    const double scale_visual = 1.0;  // already metric
    const Vec3 gravity_world(0.0, 0.0, -9.81);
    const SE3 T_cam_imu;  // identity

    auto scene = buildNoiselessEurocLikeScene(15, scale_visual, gravity_world, T_cam_imu);

    VisualInertialInitializer::Options opts;
    opts.metric_scale = true;
    VisualInertialInitializer vi(opts);

    const auto result = vi.initialize(scene.kfs);
    ASSERT_TRUE(result.converged) << "message: " << result.message;
    EXPECT_DOUBLE_EQ(result.scale, 1.0);
    EXPECT_NEAR(result.gravity_w.norm(), 9.81, 1e-3);
}
