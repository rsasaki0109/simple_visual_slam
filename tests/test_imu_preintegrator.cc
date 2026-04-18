#include <gtest/gtest.h>

#include "sensors/imu_preintegrator.h"

using namespace svslam;

namespace {

constexpr double kTolerance = 1e-9;

}

TEST(ImuPreintegratorTest, ZeroMotionGivesIdentity) {
    ImuPreintegrator p;
    // A still IMU on Earth reports +g in its z axis. With zero bias these
    // samples integrate into a pure gravity-cancelled delta (accel feeds
    // into delta_v/delta_p but rotation stays identity). We supply raw
    // samples equal to the bias (zero here) so unbiased accel is zero.
    for (int i = 0; i < 100; ++i) {
        p.integrate(Vec3::Zero(), Vec3::Zero(), 0.01);
    }
    EXPECT_NEAR(p.deltaT(), 1.0, 1e-12);
    EXPECT_LT((p.deltaR().matrix() - Mat33::Identity()).norm(), kTolerance);
    EXPECT_LT(p.deltaV().norm(), kTolerance);
    EXPECT_LT(p.deltaP().norm(), kTolerance);
}

TEST(ImuPreintegratorTest, ConstantAccelProducesExpectedDeltaVAndDeltaP) {
    ImuPreintegrator p;
    const Vec3 a_body(2.0, 0.0, 0.0);  // 2 m/s^2 along body x
    const Vec3 w_body = Vec3::Zero();
    const double dt = 0.01;
    for (int i = 0; i < 100; ++i) {
        p.integrate(a_body, w_body, dt);
    }
    EXPECT_NEAR(p.deltaT(), 1.0, 1e-12);
    // Euler integrator: delta_v = sum of a * dt = a * T
    EXPECT_NEAR(p.deltaV().x(), 2.0, 1e-6);
    // delta_p under forward Euler with R=I: p = sum_k (v_{k-1} * dt + 0.5 a dt^2)
    // Closed form: 0.5 * a * T^2 only holds for continuous integration;
    // forward Euler over 100 steps yields a slightly different number. We
    // verify the magnitude is within a small tolerance of the analytic.
    EXPECT_NEAR(p.deltaP().x(), 0.5 * 2.0 * 1.0, 2e-2);
}

TEST(ImuPreintegratorTest, PredictAppliesGravityAndInitialState) {
    ImuPreintegrator p;
    // No IMU samples => identity preintegration of duration 0.5s.
    for (int i = 0; i < 50; ++i) {
        p.integrate(Vec3::Zero(), Vec3::Zero(), 0.01);
    }
    const ImuPreintegrator::SO3 R_i;  // identity
    const Vec3 v_i(1.0, 0.0, 0.0);
    const Vec3 p_i(0.0, 0.0, 0.0);
    const Vec3 g(0.0, 0.0, -9.81);

    ImuPreintegrator::SO3 R_j;
    Vec3 v_j, p_j;
    p.predict(R_i, v_i, p_i, g, R_j, v_j, p_j);

    EXPECT_NEAR(p.deltaT(), 0.5, 1e-12);
    // R_j == R_i (no rotation)
    EXPECT_LT((R_j.matrix() - Mat33::Identity()).norm(), kTolerance);
    // v_j = v_i + g * dt
    EXPECT_NEAR(v_j.x(), 1.0, 1e-9);
    EXPECT_NEAR(v_j.z(), -9.81 * 0.5, 1e-9);
    // p_j = p_i + v_i * dt + 0.5 * g * dt^2
    EXPECT_NEAR(p_j.x(), 1.0 * 0.5, 1e-9);
    EXPECT_NEAR(p_j.z(), 0.5 * -9.81 * 0.5 * 0.5, 1e-9);
}

TEST(ImuPreintegratorTest, ResetClearsAccumulatorAndSetsBiases) {
    ImuPreintegrator p;
    p.integrate(Vec3(1, 2, 3), Vec3(0.1, 0.2, 0.3), 0.01);
    p.reset(Vec3(0.5, 0.5, 0.5), Vec3(0.01, 0.01, 0.01));
    EXPECT_DOUBLE_EQ(p.deltaT(), 0.0);
    EXPECT_LT(p.deltaV().norm(), kTolerance);
    EXPECT_LT(p.deltaP().norm(), kTolerance);
    EXPECT_DOUBLE_EQ(p.accelBias().x(), 0.5);
    EXPECT_DOUBLE_EQ(p.gyroBias().z(), 0.01);
}

TEST(ImuPreintegratorTest, BiasSubtractsFromRawMeasurement) {
    ImuPreintegrator p(Vec3(1.0, 0.0, 0.0), Vec3::Zero());
    // Raw accel equal to bias => unbiased accel is zero, no motion.
    for (int i = 0; i < 50; ++i) {
        p.integrate(Vec3(1.0, 0.0, 0.0), Vec3::Zero(), 0.01);
    }
    EXPECT_LT(p.deltaV().norm(), kTolerance);
    EXPECT_LT(p.deltaP().norm(), kTolerance);
}
