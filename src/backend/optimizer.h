#pragma once

#include "core/common.h"
#include "core/map.h"
#include "core/frame.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace svslam {

// Depth prior cost function: constrains z-component of point in camera frame
struct DepthPriorError {
    DepthPriorError(double observed_depth, double fx, double fy, double cx, double cy,
                    double observed_u, double observed_v, double weight)
        : observed_depth(observed_depth), fx(fx), fy(fy), cx(cx), cy(cy),
          observed_u(observed_u), observed_v(observed_v), weight(weight) {}

    template <typename T>
    bool operator()(const T* const camera_pose,  // [tx, ty, tz, qw, qx, qy, qz]
                    const T* const point,         // [x, y, z]
                    T* residuals) const {
        // Transform point from world to camera frame
        T p[3];
        ceres::QuaternionRotatePoint(camera_pose + 3, point, p);
        p[0] += camera_pose[0];
        p[1] += camera_pose[1];
        p[2] += camera_pose[2];

        // Depth residual: predicted z in camera frame vs observed depth
        residuals[0] = T(weight) * (p[2] - T(observed_depth));
        return true;
    }

    static ceres::CostFunction* Create(double observed_depth, double fx, double fy,
                                        double cx, double cy,
                                        double observed_u, double observed_v,
                                        double weight) {
        return new ceres::AutoDiffCostFunction<DepthPriorError, 1, 7, 3>(
            new DepthPriorError(observed_depth, fx, fy, cx, cy,
                                observed_u, observed_v, weight));
    }

    double observed_depth;
    double fx, fy, cx, cy;
    double observed_u, observed_v;
    double weight;
};

// IMU preintegration residual between consecutive keyframes (Stage 0c.c/a).
//
// Forster-style 6-DoF (position + velocity) residual consuming frozen
// preintegration deltas (delta_p, delta_v, dt) computed between KF_i and
// KF_j at a reference accelerometer bias (bias_accel_ref). First-order
// bias Jacobians let BA re-estimate the accel bias without re-running
// preintegration:
//   J_p_ba = -0.5 * dt^2 * I
//   J_v_ba = -dt * I
// Gyro-bias effects on delta_p/delta_v are dropped (they require rotation
// Jacobians); the gyro bias is instead shaped by anchor and random-walk
// priors. Rotation alignment between camera and IMU is approximated as
// identity for scaffolding.
struct VelocityPreintegrationError {
    VelocityPreintegrationError(const Vec3& delta_p,
                                const Vec3& delta_v,
                                const Vec3& bias_accel_ref,
                                double dt,
                                const Vec3& gravity_w,
                                const Eigen::Quaterniond& q_cam_imu,
                                const Vec3& t_cam_imu,
                                double pos_weight,
                                double vel_weight)
        : dt_(dt), pos_weight_(pos_weight), vel_weight_(vel_weight) {
        dp_[0] = delta_p.x(); dp_[1] = delta_p.y(); dp_[2] = delta_p.z();
        dv_[0] = delta_v.x(); dv_[1] = delta_v.y(); dv_[2] = delta_v.z();
        g_[0] = gravity_w.x(); g_[1] = gravity_w.y(); g_[2] = gravity_w.z();
        ba_ref_[0] = bias_accel_ref.x();
        ba_ref_[1] = bias_accel_ref.y();
        ba_ref_[2] = bias_accel_ref.z();
        // T_cam_imu rotation: q_cb s.t. p_cam = q_cb * p_body. For the
        // preintegration residual we need q_wb = q_wc * q_cb, which is the
        // body's world rotation — body-frame deltas then rotate into world
        // via p_world = q_wb * p_body.
        q_cb_[0] = q_cam_imu.w();
        q_cb_[1] = q_cam_imu.x();
        q_cb_[2] = q_cam_imu.y();
        q_cb_[3] = q_cam_imu.z();
        // camera origin expressed in body frame (t_bc). For T_cb = [q_cb |
        // t_cb] (body→cam), T_bc = inverse gives t_bc = -q_cb^T * t_cb.
        const Eigen::Quaterniond q_bc = q_cam_imu.conjugate();
        const Vec3 t_bc = -(q_bc * t_cam_imu);
        t_bc_[0] = t_bc.x();
        t_bc_[1] = t_bc.y();
        t_bc_[2] = t_bc.z();
    }

    template <typename T>
    bool operator()(const T* const pose_i,        // T_cw_i
                    const T* const pose_j,        // T_cw_j
                    const T* const vel_i,         // world-frame velocity of IMU body at KF_i
                    const T* const vel_j,         // world-frame velocity of IMU body at KF_j
                    const T* const bias_accel_i,  // accel bias at KF_i (m/s^2)
                    T* residuals) const {
        // q_wc_i = conjugate(q_cw_i)
        const T q_wc_i[4] = {pose_i[3], -pose_i[4], -pose_i[5], -pose_i[6]};
        const T neg_t_i[3] = {-pose_i[0], -pose_i[1], -pose_i[2]};
        T p_wc_i[3];
        ceres::QuaternionRotatePoint(q_wc_i, neg_t_i, p_wc_i);

        const T q_wc_j[4] = {pose_j[3], -pose_j[4], -pose_j[5], -pose_j[6]};
        const T neg_t_j[3] = {-pose_j[0], -pose_j[1], -pose_j[2]};
        T p_wc_j[3];
        ceres::QuaternionRotatePoint(q_wc_j, neg_t_j, p_wc_j);

        // q_wb_i = q_wc_i * q_cb (body-in-world rotation at KF_i).
        const T q_cb[4] = {T(q_cb_[0]), T(q_cb_[1]), T(q_cb_[2]), T(q_cb_[3])};
        T q_wb_i[4];
        ceres::QuaternionProduct(q_wc_i, q_cb, q_wb_i);
        T q_wb_j[4];
        ceres::QuaternionProduct(q_wc_j, q_cb, q_wb_j);

        // p_wb = p_wc - R_wb * t_bc (camera origin in body frame → world).
        const T t_bc[3] = {T(t_bc_[0]), T(t_bc_[1]), T(t_bc_[2])};
        T R_wb_t_bc_i[3];
        ceres::QuaternionRotatePoint(q_wb_i, t_bc, R_wb_t_bc_i);
        T R_wb_t_bc_j[3];
        ceres::QuaternionRotatePoint(q_wb_j, t_bc, R_wb_t_bc_j);
        const T p_wb_i[3] = {p_wc_i[0] - R_wb_t_bc_i[0],
                             p_wc_i[1] - R_wb_t_bc_i[1],
                             p_wc_i[2] - R_wb_t_bc_i[2]};
        const T p_wb_j[3] = {p_wc_j[0] - R_wb_t_bc_j[0],
                             p_wc_j[1] - R_wb_t_bc_j[1],
                             p_wc_j[2] - R_wb_t_bc_j[2]};

        const T dt = T(dt_);
        const T half_dt2 = T(0.5) * dt * dt;

        // First-order accel-bias correction in the body-at-i frame.
        const T dba[3] = {
            bias_accel_i[0] - T(ba_ref_[0]),
            bias_accel_i[1] - T(ba_ref_[1]),
            bias_accel_i[2] - T(ba_ref_[2])
        };
        const T dp_corr[3] = {
            T(dp_[0]) - half_dt2 * dba[0],
            T(dp_[1]) - half_dt2 * dba[1],
            T(dp_[2]) - half_dt2 * dba[2]
        };
        const T dv_corr[3] = {
            T(dv_[0]) - dt * dba[0],
            T(dv_[1]) - dt * dba[1],
            T(dv_[2]) - dt * dba[2]
        };

        // Rotate body-frame (bias-corrected) deltas into world via R_wb_i.
        T R_dp[3];
        ceres::QuaternionRotatePoint(q_wb_i, dp_corr, R_dp);
        T R_dv[3];
        ceres::QuaternionRotatePoint(q_wb_i, dv_corr, R_dv);

        residuals[0] = T(pos_weight_) *
                       ((p_wb_j[0] - p_wb_i[0]) - vel_i[0] * dt - T(g_[0]) * half_dt2 - R_dp[0]);
        residuals[1] = T(pos_weight_) *
                       ((p_wb_j[1] - p_wb_i[1]) - vel_i[1] * dt - T(g_[1]) * half_dt2 - R_dp[1]);
        residuals[2] = T(pos_weight_) *
                       ((p_wb_j[2] - p_wb_i[2]) - vel_i[2] * dt - T(g_[2]) * half_dt2 - R_dp[2]);
        residuals[3] = T(vel_weight_) *
                       (vel_j[0] - vel_i[0] - T(g_[0]) * dt - R_dv[0]);
        residuals[4] = T(vel_weight_) *
                       (vel_j[1] - vel_i[1] - T(g_[1]) * dt - R_dv[1]);
        residuals[5] = T(vel_weight_) *
                       (vel_j[2] - vel_i[2] - T(g_[2]) * dt - R_dv[2]);
        return true;
    }

    static ceres::CostFunction* Create(const Vec3& delta_p,
                                       const Vec3& delta_v,
                                       const Vec3& bias_accel_ref,
                                       double dt,
                                       const Vec3& gravity_w,
                                       const Eigen::Quaterniond& q_cam_imu,
                                       const Vec3& t_cam_imu,
                                       double pos_weight,
                                       double vel_weight) {
        return new ceres::AutoDiffCostFunction<VelocityPreintegrationError, 6, 7, 7, 3, 3, 3>(
            new VelocityPreintegrationError(delta_p, delta_v, bias_accel_ref, dt,
                                            gravity_w, q_cam_imu, t_cam_imu,
                                            pos_weight, vel_weight));
    }

    double dp_[3];
    double dv_[3];
    double g_[3];
    double ba_ref_[3];
    double q_cb_[4];  // w, x, y, z
    double t_bc_[3];
    double dt_;
    double pos_weight_;
    double vel_weight_;
};

// Zero-anchor prior on an IMU bias (accel or gyro). Keeps the bias near
// origin with a soft pull, preventing it from drifting when only the
// weak preintegration residual applies.
struct BiasAnchorError {
    explicit BiasAnchorError(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const bias, T* residuals) const {
        residuals[0] = T(weight_) * bias[0];
        residuals[1] = T(weight_) * bias[1];
        residuals[2] = T(weight_) * bias[2];
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<BiasAnchorError, 3, 3>(
            new BiasAnchorError(weight));
    }

    double weight_;
};

// Random-walk prior between consecutive keyframes' biases. Enforces the
// "bias varies slowly" assumption; smaller sigma → more coupling.
struct BiasRandomWalkError {
    explicit BiasRandomWalkError(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const bias_i,
                    const T* const bias_j,
                    T* residuals) const {
        residuals[0] = T(weight_) * (bias_j[0] - bias_i[0]);
        residuals[1] = T(weight_) * (bias_j[1] - bias_i[1]);
        residuals[2] = T(weight_) * (bias_j[2] - bias_i[2]);
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<BiasRandomWalkError, 3, 3, 3>(
            new BiasRandomWalkError(weight));
    }

    double weight_;
};

// Loose IMU velocity prior between consecutive keyframes.
//
// Constrains the camera position delta in the world frame between two
// keyframes to match v_i_world * dt (ignoring 0.5 g dt^2 since sub-0.1s
// keyframe gaps give <5 cm gravity drop, well inside the loose sigma).
//
// Velocity itself is not a parameter block here: the IMU preintegration in
// Tracking writes KF_i->velocity_ ahead of BA and BA only uses it as a soft
// prior on pose translations. This is the "loose prior, not tight VIO"
// scaffolding for Stage 0b.
struct VelocityDeltaPriorError {
    VelocityDeltaPriorError(const Vec3& v_i_world, double dt, double weight)
        : dt_(dt), weight_(weight) {
        v_wx_ = v_i_world.x();
        v_wy_ = v_i_world.y();
        v_wz_ = v_i_world.z();
    }

    template <typename T>
    bool operator()(const T* const pose_i,  // [tx, ty, tz, qw, qx, qy, qz] for T_cw_i
                    const T* const pose_j,
                    T* residuals) const {
        // p_wc = -R_cw^T * t_cw = rotate(-t_cw) by the inverse (conjugate) of q_cw.
        const T q_i_inv[4] = {pose_i[3], -pose_i[4], -pose_i[5], -pose_i[6]};
        const T neg_t_i[3] = {-pose_i[0], -pose_i[1], -pose_i[2]};
        T p_wc_i[3];
        ceres::QuaternionRotatePoint(q_i_inv, neg_t_i, p_wc_i);

        const T q_j_inv[4] = {pose_j[3], -pose_j[4], -pose_j[5], -pose_j[6]};
        const T neg_t_j[3] = {-pose_j[0], -pose_j[1], -pose_j[2]};
        T p_wc_j[3];
        ceres::QuaternionRotatePoint(q_j_inv, neg_t_j, p_wc_j);

        const T dp_x = p_wc_j[0] - p_wc_i[0];
        const T dp_y = p_wc_j[1] - p_wc_i[1];
        const T dp_z = p_wc_j[2] - p_wc_i[2];

        residuals[0] = T(weight_) * (dp_x - T(v_wx_) * T(dt_));
        residuals[1] = T(weight_) * (dp_y - T(v_wy_) * T(dt_));
        residuals[2] = T(weight_) * (dp_z - T(v_wz_) * T(dt_));
        return true;
    }

    static ceres::CostFunction* Create(const Vec3& v_i_world, double dt, double weight) {
        return new ceres::AutoDiffCostFunction<VelocityDeltaPriorError, 3, 7, 7>(
            new VelocityDeltaPriorError(v_i_world, dt, weight));
    }

    double v_wx_, v_wy_, v_wz_;
    double dt_;
    double weight_;
};

// Gravity prior cost function: constrains roll/pitch by requiring that
// R_cw * gravity_world ≈ gravity_camera (measured by accelerometer)
// gravity_world = [0, 0, -1] after gravity alignment
// This leaves yaw unconstrained (1 DOF free)
struct GravityPriorError {
    GravityPriorError(double gx_cam, double gy_cam, double gz_cam, double weight)
        : gx_cam(gx_cam), gy_cam(gy_cam), gz_cam(gz_cam), weight(weight) {}

    template <typename T>
    bool operator()(const T* const camera_pose,  // [tx, ty, tz, qw, qx, qy, qz]
                    T* residuals) const {
        // World gravity direction (after gravity alignment): [0, 0, -1]
        const T g_world[3] = {T(0), T(0), T(-1)};

        // Rotate world gravity to camera frame: g_cam_pred = R_cw * g_world
        T g_cam_pred[3];
        ceres::QuaternionRotatePoint(camera_pose + 3, g_world, g_cam_pred);

        // Residual: predicted vs observed gravity in camera frame
        residuals[0] = T(weight) * (g_cam_pred[0] - T(gx_cam));
        residuals[1] = T(weight) * (g_cam_pred[1] - T(gy_cam));
        residuals[2] = T(weight) * (g_cam_pred[2] - T(gz_cam));
        return true;
    }

    static ceres::CostFunction* Create(double gx_cam, double gy_cam, double gz_cam, double weight) {
        return new ceres::AutoDiffCostFunction<GravityPriorError, 3, 7>(
            new GravityPriorError(gx_cam, gy_cam, gz_cam, weight));
    }

    double gx_cam, gy_cam, gz_cam;
    double weight;
};

class Optimizer {
public:
    struct PoseGraphEdge {
        Keyframe::Ptr from;
        Keyframe::Ptr to;
        Sim3 relative_pose;
        double translation_weight = 1.0;
        double rotation_weight = 1.0;
        double scale_weight = 1.0;
    };

    // Local Bundle Adjustment
    // Optimize a keyframe and its neighbors, and observed landmarks
    static void bundleAdjustment(const std::vector<Keyframe::Ptr>& keyframes, 
                                 const std::vector<Landmark::Ptr>& landmarks,
                                 int iterations = 10);

    static void poseGraphOptimization(Map::Ptr map,
                                      const std::vector<PoseGraphEdge>& loop_edges,
                                      int iterations = 50,
                                      bool fix_scale = false);

    static void globalBundleAdjustment(Map::Ptr map, int iterations = 10);
                                 
    // Pose optimization only (e.g. for tracking)
    static int poseOptimization(Frame::Ptr frame);
};

}
