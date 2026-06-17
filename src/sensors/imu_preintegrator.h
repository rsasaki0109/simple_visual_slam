#pragma once

#include <sophus/so3.hpp>

#include "core/common.h"
#include "sensors/imu.h"

namespace svslam {

// Minimal Forster-style IMU preintegration.
// Accumulates delta_R, delta_v, delta_p over an interval [i, j] given the
// IMU biases at i. Supports predicting the keyframe-j state from the
// keyframe-i state plus the gravity vector in the world frame.
//
// This class is intentionally scaffolding-level: bias Jacobians and noise
// covariance propagation are left out for a follow-up commit when a VIO
// residual is actually wired into BA.
class ImuPreintegrator {
public:
    using SO3 = Sophus::SO3d;

    ImuPreintegrator() { reset(Vec3::Zero(), Vec3::Zero()); }
    ImuPreintegrator(const Vec3& accel_bias, const Vec3& gyro_bias) {
        reset(accel_bias, gyro_bias);
    }

    // Discard any accumulated state and re-anchor with the given biases.
    void reset(const Vec3& accel_bias, const Vec3& gyro_bias) {
        delta_R_ = SO3();
        delta_v_ = Vec3::Zero();
        delta_p_ = Vec3::Zero();
        dt_ = 0.0;
        accel_bias_ = accel_bias;
        gyro_bias_ = gyro_bias;
    }

    // Integrate one IMU sample over a dt second interval using midpoint-free
    // (Euler) integration in the i-frame. Callers are responsible for
    // selecting dt from the IMU timestamps.
    void integrate(const Vec3& accel, const Vec3& gyro, double dt) {
        if (!(dt > 0.0)) {
            return;
        }
        const Vec3 a_unbiased = accel - accel_bias_;
        const Vec3 w_unbiased = gyro - gyro_bias_;

        // Position and velocity increments must be evaluated BEFORE updating
        // delta_R_ so the integrator uses the start-of-interval rotation.
        const Vec3 accel_i = delta_R_ * a_unbiased;
        delta_p_ += delta_v_ * dt + 0.5 * accel_i * dt * dt;
        delta_v_ += accel_i * dt;

        // Right-multiply rotation with Exp(w dt).
        delta_R_ = delta_R_ * SO3::exp(w_unbiased * dt);

        dt_ += dt;
    }

    // Predict keyframe-j state from keyframe-i state (world frame) and
    // gravity vector (world frame, m/s^2; e.g. Vec3(0, 0, -9.81) for TUM).
    void predict(const SO3& R_i,
                 const Vec3& v_i,
                 const Vec3& p_i,
                 const Vec3& gravity_w,
                 SO3& R_j,
                 Vec3& v_j,
                 Vec3& p_j) const {
        R_j = R_i * delta_R_;
        v_j = v_i + gravity_w * dt_ + R_i * delta_v_;
        p_j = p_i + v_i * dt_ + 0.5 * gravity_w * dt_ * dt_ + R_i * delta_p_;
    }

    const SO3& deltaR() const { return delta_R_; }
    const Vec3& deltaV() const { return delta_v_; }
    const Vec3& deltaP() const { return delta_p_; }
    double deltaT() const { return dt_; }
    const Vec3& accelBias() const { return accel_bias_; }
    const Vec3& gyroBias() const { return gyro_bias_; }

private:
    SO3 delta_R_;
    Vec3 delta_v_ = Vec3::Zero();
    Vec3 delta_p_ = Vec3::Zero();
    double dt_ = 0.0;
    Vec3 accel_bias_ = Vec3::Zero();
    Vec3 gyro_bias_ = Vec3::Zero();
};

}  // namespace svslam
