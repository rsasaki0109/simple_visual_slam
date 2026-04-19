#include "tracking/visual_inertial_initializer.h"

#include <cmath>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace svslam {

namespace {

// Small helper to make a VI-init Result carrying an error message and
// leaving the rest of the fields at their defaults.
VisualInertialInitializer::Result makeFailure(const std::string& msg) {
    VisualInertialInitializer::Result r;
    r.converged = false;
    r.message = msg;
    return r;
}

// Visual rotation R_wb = R_wc * R_cb, computed from a KF's current pose
// T_cw and the camera-from-IMU extrinsic stored with the span.
Eigen::Matrix3d bodyRotationInWorld(const Keyframe::Ptr& kf,
                                    const Sophus::SO3d& R_cb) {
    // T_cw_ maps world → camera; inverse gives world-from-camera pose.
    const Sophus::SO3d R_wc = kf->T_cw_.so3().inverse();
    return (R_wc * R_cb).matrix();
}

// Right-Jacobian inverse times a small rotation vector — used for
// approximating the first-order gyro-bias Jacobian of delta_R.
// J_R^{-1}(phi) ≈ I + 0.5 * [phi]_x for small phi; for small bias updates
// it's accurate enough to fold into the closed-form bias solve. For our
// purposes we just assume J ≈ -I * dt, which is the textbook result when
// the rotation change is dominated by the bias term (see Forster et al.).

}  // namespace

VisualInertialInitializer::Result VisualInertialInitializer::initialize(
    const std::vector<Keyframe::Ptr>& keyframes) const {
    if (keyframes.size() < 3) {
        return makeFailure("need at least 3 keyframes");
    }

    // Basic sanity + topology checks. Every KF past the first one must
    // carry a valid span whose `from_kf_id` points at the preceding KF.
    // We pull R_cb / t_cb from the first valid span so the rest of the
    // algorithm can use a single extrinsic; the initializer is not
    // designed to handle per-KF extrinsic variation.
    Sophus::SO3d R_cb_global;
    Eigen::Vector3d t_cb_global = Eigen::Vector3d::Zero();
    bool extrinsic_ready = false;

    for (std::size_t i = 1; i < keyframes.size(); ++i) {
        const auto& kf_i = keyframes[i - 1];
        const auto& kf_j = keyframes[i];
        if (!kf_i || !kf_j) {
            return makeFailure("null keyframe in window");
        }
        if (!kf_j->prev_imu_span_ || !kf_j->prev_imu_span_->valid) {
            return makeFailure("missing / invalid prev_imu_span_");
        }
        if (kf_j->prev_imu_span_->from_kf_id != kf_i->id_) {
            return makeFailure("span predecessor mismatch");
        }
        if (!(kf_j->prev_imu_span_->dt > 0.0)) {
            return makeFailure("non-positive span dt");
        }
        if (!extrinsic_ready) {
            R_cb_global = kf_j->prev_imu_span_->T_cam_imu.so3();
            t_cb_global = kf_j->prev_imu_span_->T_cam_imu.translation();
            extrinsic_ready = true;
        }
    }

    if (!extrinsic_ready) {
        return makeFailure("no extrinsic available");
    }

    // Cumulative duration guard — short windows leave scale / gravity
    // poorly observable since the 0.5*g*dt^2 term dominates the linear
    // system only when dt is not microscopic.
    double total_dt = 0.0;
    for (std::size_t i = 1; i < keyframes.size(); ++i) {
        total_dt += keyframes[i]->prev_imu_span_->dt;
    }
    if (total_dt < options_.min_total_dt_seconds) {
        return makeFailure("insufficient cumulative span duration");
    }

    // ---------- Stage 1: closed-form gyro bias ----------
    //
    // Preintegration gave us delta_R at bias_gyro = span.bias_gyro. The
    // visually observed rotation between KF_i and KF_j, expressed in the
    // body frame, is:  dR_obs = R_wb_i^T * R_wb_j.
    // Approximating the bias-update Jacobian as -I * dt, the residual
    // r_i  = Log((delta_R)^T * dR_obs) ≈ -dbg * dt, stacked across all
    // pairs gives a simple diagonal-weighted 3×3 system.
    Eigen::Vector3d estimated_gyro_bias = keyframes[1]->prev_imu_span_->bias_gyro;
    double rotation_residual_rms = 0.0;
    {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
        double residual_sum_sq = 0.0;
        int pair_count = 0;
        for (std::size_t i = 1; i < keyframes.size(); ++i) {
            const auto& span = *keyframes[i]->prev_imu_span_;
            const Eigen::Matrix3d R_wb_i = bodyRotationInWorld(keyframes[i - 1], R_cb_global);
            const Eigen::Matrix3d R_wb_j = bodyRotationInWorld(keyframes[i], R_cb_global);
            const Eigen::Matrix3d dR_obs = R_wb_i.transpose() * R_wb_j;
            const Eigen::Matrix3d R_err = span.delta_R.matrix().transpose() * dR_obs;
            const Eigen::Vector3d r = Sophus::SO3d(Eigen::Quaterniond(R_err)).log();
            // J = -I * dt (first order); so dbg satisfies J * dbg = r
            //   => dbg = -r / dt; stacked: H = sum(dt^2 * I), rhs = sum(dt * -r)
            // Equivalent to H * dbg = rhs with H = sum(dt^2) * I.
            const double dt = span.dt;
            H += dt * dt * Eigen::Matrix3d::Identity();
            rhs += -dt * r;
            residual_sum_sq += r.squaredNorm();
            ++pair_count;
        }
        if (pair_count == 0) {
            return makeFailure("gyro bias system degenerate");
        }
        // Tikhonov regularization toward zero. Short EuRoC windows leave
        // the bias underdetermined, so an unregularized solve blows up
        // (observed |bg| ~ 0.15 rad/s against a true ~0.004 rad/s). The
        // prior pulls weakly toward zero while still letting real biases
        // show through.
        if (options_.gyro_bias_prior_sigma > 0.0) {
            const double lambda = 1.0 /
                (options_.gyro_bias_prior_sigma * options_.gyro_bias_prior_sigma);
            H += lambda * Eigen::Matrix3d::Identity();
            // rhs += lambda * 0 (zero prior mean).
        }
        if (H.determinant() < 1e-18) {
            return makeFailure("gyro bias system degenerate");
        }
        Eigen::Vector3d dbg = H.ldlt().solve(rhs);
        if (!dbg.allFinite()) {
            return makeFailure("gyro bias solve non-finite");
        }
        // Hard cap — if the regularized solve still lands outside a sane
        // range, fall back to zero. Better to let the BA bias blocks
        // discover the true bias than to seed preintegration with a wildly
        // wrong delta_R correction.
        if (options_.gyro_bias_magnitude_cap > 0.0 &&
            dbg.cwiseAbs().maxCoeff() > options_.gyro_bias_magnitude_cap) {
            dbg.setZero();
        }
        estimated_gyro_bias = keyframes[1]->prev_imu_span_->bias_gyro + dbg;
        rotation_residual_rms =
            pair_count > 0 ? std::sqrt(residual_sum_sq / pair_count) : 0.0;
    }

    // ---------- Stage 2: scale + gravity + per-KF velocities ----------
    //
    // For each pair (i, j), use the Forster preintegration equations with
    // accel_bias fixed at span.bias_accel (kept as reference — the linear
    // fit can't tease scale + bias apart in a short window, so we leave
    // bias refinement to the BA that consumes this bootstrap):
    //
    //   s * (p_wb_j - p_wb_i) = v_i * dt_ij + 0.5 * g_w * dt_ij^2
    //                         + R_wb_i * delta_p
    //
    //   v_j - v_i = g_w * dt_ij + R_wb_i * delta_v
    //
    // Unknowns: x = [v_0 | v_1 | ... | v_{N-1} | g_w | s]
    //              size = 3*N + 3 + 1  (when `metric_scale` is false)
    //              size = 3*N + 3       (when `metric_scale` is true)
    const std::size_t N = keyframes.size();
    const bool solve_scale = !options_.metric_scale;
    const bool gravity_fixed = options_.assume_gravity_w.allFinite() &&
                               options_.assume_gravity_w.norm() > 1e-6;
    const Eigen::Vector3d g_fixed =
        gravity_fixed ? options_.assume_gravity_w : Eigen::Vector3d::Zero();
    const std::size_t vel_dim = 3 * N;
    const std::size_t grav_off = vel_dim;  // only used when !gravity_fixed
    const std::size_t scale_off = gravity_fixed ? vel_dim : (vel_dim + 3);
    const std::size_t unknowns =
        vel_dim + (gravity_fixed ? 0 : 3) + (solve_scale ? 1 : 0);
    const bool add_scale_prior = solve_scale && options_.scale_prior_weight > 0.0;
    const std::size_t rows = 6 * (N - 1) + (add_scale_prior ? 1 : 0);

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, unknowns);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

    // The visual mono rescaling applies to *camera* positions (that's what
    // median-depth normalisation touches), not to the IMU lever-arm t_bc.
    // Using camera-frame positions in the LSQ avoids coupling scale into
    // the lever-arm term:
    //   p_wc_metric = p_wb_metric + R_wb * t_bc
    // ⇒ p_wb_metric_j - p_wb_metric_i
    //   = (p_wc_metric_j - p_wc_metric_i) - (R_wb_j - R_wb_i) * t_bc
    //   = s * (p_wc_visual_j - p_wc_visual_i) - (R_wb_j - R_wb_i) * t_bc
    // Substituting into the body-frame Forster equation and re-arranging,
    // the scale coefficient is cleanly (p_wc_j - p_wc_i) only.
    const Eigen::Vector3d t_bc = -(R_cb_global.inverse() * t_cb_global);

    for (std::size_t i = 1; i < N; ++i) {
        const auto& span = *keyframes[i]->prev_imu_span_;
        const double dt = span.dt;
        const double half_dt2 = 0.5 * dt * dt;

        const Eigen::Matrix3d R_wb_i = bodyRotationInWorld(keyframes[i - 1], R_cb_global);
        const Eigen::Matrix3d R_wb_j = bodyRotationInWorld(keyframes[i], R_cb_global);

        // Camera-in-world positions (visual frame — these are what the
        // downstream rescale multiplies by s).
        const Eigen::Vector3d p_wc_i = keyframes[i - 1]->T_cw_.inverse().translation();
        const Eigen::Vector3d p_wc_j = keyframes[i]->T_cw_.inverse().translation();

        const Eigen::Vector3d R_dp = R_wb_i * span.delta_p;
        const Eigen::Vector3d R_dv = R_wb_i * span.delta_v;
        const Eigen::Vector3d lever_term = (R_wb_j - R_wb_i) * t_bc;

        const std::size_t row_pos = 6 * (i - 1);
        const std::size_t row_vel = row_pos + 3;

        // Position: s*(p_wc_j - p_wc_i) - dt*v_i - 0.5*dt^2*g
        //         = R_wb_i*delta_p + (R_wb_j - R_wb_i) * t_bc
        A.block<3, 3>(row_pos, 3 * (i - 1)) = -dt * Eigen::Matrix3d::Identity();  // v_i
        Eigen::Vector3d rhs_pos = R_dp + lever_term;
        if (gravity_fixed) {
            // Bake the fixed gravity term into the RHS.
            rhs_pos += half_dt2 * g_fixed;
        } else {
            A.block<3, 3>(row_pos, grav_off) = -half_dt2 * Eigen::Matrix3d::Identity();  // g
        }
        if (solve_scale) {
            A.block<3, 1>(row_pos, scale_off) = p_wc_j - p_wc_i;  // s
            b.segment<3>(row_pos) = rhs_pos;
        } else {
            b.segment<3>(row_pos) = rhs_pos - (p_wc_j - p_wc_i);
        }

        // Velocity: v_j - v_i - dt*g = R_wb_i*delta_v (no scale coupling)
        A.block<3, 3>(row_vel, 3 * i) = Eigen::Matrix3d::Identity();     // v_j
        A.block<3, 3>(row_vel, 3 * (i - 1)) -= Eigen::Matrix3d::Identity();  // -v_i
        Eigen::Vector3d rhs_vel = R_dv;
        if (gravity_fixed) {
            rhs_vel += dt * g_fixed;
        } else {
            A.block<3, 3>(row_vel, grav_off) = -dt * Eigen::Matrix3d::Identity();
        }
        b.segment<3>(row_vel) = rhs_vel;
    }

    if (add_scale_prior) {
        const std::size_t row = 6 * (N - 1);
        A(row, scale_off) = options_.scale_prior_weight;
        b(row) = options_.scale_prior_weight * options_.scale_prior;
    }

    // Least squares solve. Bounded scale check afterwards.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd x = svd.solve(b);
    if (!x.allFinite()) {
        return makeFailure("linear solve non-finite");
    }

    if (std::getenv("SVSLAM_VIO_INIT_DEBUG")) {
        std::cerr << "[VI init debug] N=" << N << " unknowns=" << unknowns
                  << " rows=" << rows
                  << " gravity_fixed=" << gravity_fixed
                  << " solve_scale=" << solve_scale << std::endl;
        std::cerr << "  T_cb (cam from imu) R=\n" << R_cb_global.matrix()
                  << "\n  t_cb=" << t_cb_global.transpose() << std::endl;
        std::cerr << "  KF IDs: ";
        for (std::size_t i = 0; i < std::min<std::size_t>(N, 15); ++i) {
            std::cerr << keyframes[i]->id_ << " ";
        }
        std::cerr << std::endl;
        for (std::size_t i = 1; i < std::min<std::size_t>(N, 4); ++i) {
            const auto& span = *keyframes[i]->prev_imu_span_;
            const Eigen::Vector3d p_wc_i = keyframes[i - 1]->T_cw_.inverse().translation();
            const Eigen::Vector3d p_wc_j = keyframes[i]->T_cw_.inverse().translation();
            const Eigen::Matrix3d R_wb_i = bodyRotationInWorld(keyframes[i - 1], R_cb_global);
            const Eigen::Matrix3d R_wc_i = keyframes[i - 1]->T_cw_.inverse().so3().matrix();
            const Eigen::Vector3d R_dp = R_wb_i * span.delta_p;
            const Eigen::Vector3d g_in_body_est = R_wb_i.transpose() * Eigen::Vector3d(0, 0, -9.81);
            std::cerr << "  pair " << (i - 1) << "->" << i
                      << " dt=" << span.dt
                      << "\n    dp_wc_visual=" << (p_wc_j - p_wc_i).transpose()
                      << "\n    R_dp_metric =" << R_dp.transpose()
                      << "\n    dp_body     =" << span.delta_p.transpose()
                      << "\n    g_in_body   =" << g_in_body_est.transpose()
                      << "\n    R_wc_i.det  =" << R_wc_i.determinant()
                      << std::endl;
        }
        Eigen::Vector3d g_sol = gravity_fixed ? g_fixed
                                              : Eigen::Vector3d(x[grav_off], x[grav_off+1], x[grav_off+2]);
        std::cerr << "  solution scale=" << (solve_scale ? x[scale_off] : 1.0)
                  << "\n  velocity_0=" << Eigen::Vector3d(x[0], x[1], x[2]).transpose()
                  << "\n  gravity_world=" << g_sol.transpose()
                  << " (|g|=" << g_sol.norm() << ")"
                  << std::endl;
    }

    const Eigen::VectorXd residual = A * x - b;
    const double linear_rms = std::sqrt(residual.squaredNorm() / std::max<std::size_t>(1, rows));

    Result out;
    out.velocities.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        out.velocities.emplace_back(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    }
    out.gravity_w = gravity_fixed
        ? g_fixed
        : Eigen::Vector3d(x[grav_off], x[grav_off + 1], x[grav_off + 2]);
    out.scale = solve_scale ? x[scale_off] : 1.0;
    out.gyro_bias = estimated_gyro_bias;
    out.accel_bias = Eigen::Vector3d::Zero();  // first-pass pin
    out.rotation_residual_rms = rotation_residual_rms;
    out.linear_residual_rms = linear_rms;

    // Sanity gates.
    if (solve_scale) {
        const double s = out.scale;
        const double ratio = s > 0.0 ? s : -s;
        if (!(s > 0.0)) {
            out.message = "non-positive scale";
            out.converged = false;
            return out;
        }
        if (ratio > options_.max_scale_ratio_deviation ||
            1.0 / ratio > options_.max_scale_ratio_deviation) {
            out.message = "scale outside tolerance";
            out.converged = false;
            return out;
        }
    }

    const double g_norm = out.gravity_w.norm();
    if (!(g_norm > 1.0)) {
        out.message = "gravity vector near zero";
        out.converged = false;
        return out;
    }

    // Rescale gravity to the expected magnitude. This preserves the
    // direction while letting the downstream BA math use |g| = 9.81
    // without tracking a per-run scale.
    out.gravity_w *= (options_.gravity_magnitude / g_norm);

    // Propagate the scale into the velocities so the caller can consume a
    // coherent (metric) trajectory. (p_j - p_i) * s accounts for scale in
    // positions, but v_i was solved in metric units directly — no
    // rescaling needed. Scale affects positions only; velocities are
    // already in m/s.

    if (!std::isfinite(linear_rms) || linear_rms > options_.linear_residual_rms_max) {
        out.message = "linear residual too large";
        out.converged = false;
        return out;
    }

    // Final convergence gate — rotation residual failure also prevents
    // success.
    if (rotation_residual_rms > options_.rotation_residual_rms_max) {
        out.message = "rotation residual too large";
        out.converged = false;
        return out;
    }

    out.converged = true;
    if (out.message.empty()) {
        std::ostringstream os;
        os << "VI init OK (scale=" << out.scale
           << " |g|=" << options_.gravity_magnitude
           << " rot_rms=" << out.rotation_residual_rms
           << " lin_rms=" << out.linear_residual_rms
           << ")";
        out.message = os.str();
    }
    return out;
}

void VisualInertialInitializer::applyGyroBiasCorrectionToSpans(
    const std::vector<Keyframe::Ptr>& keyframes,
    const Result& result) const {
    if (!result.converged) return;
    if (keyframes.size() < 2) return;

    for (std::size_t i = 1; i < keyframes.size(); ++i) {
        auto& kf = keyframes[i];
        if (!kf || !kf->prev_imu_span_ || !kf->prev_imu_span_->valid) continue;
        auto& span = *kf->prev_imu_span_;

        // First-order Forster correction: delta_R(bg) ≈ delta_R_ref *
        // Exp(-J_r * (bg_new - bg_ref) * dt). For short inter-KF intervals
        // we approximate J_r = I, which keeps the update a 3-vector tangent
        // rotation that's safe to compose with Sophus::SO3d.
        const Eigen::Vector3d dbg = result.gyro_bias - span.bias_gyro;
        if (!dbg.allFinite()) continue;
        const Eigen::Vector3d phi = -dbg * span.dt;
        span.delta_R = span.delta_R * Sophus::SO3d::exp(phi);
        span.bias_gyro = result.gyro_bias;
    }
}

}  // namespace svslam
