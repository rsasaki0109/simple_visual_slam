#pragma once

#include <memory>
#include <vector>

#include "core/common.h"
#include "core/keyframe.h"

namespace svslam {

// Visual-Inertial Initializer (VIO Stage 0c.e).
//
// Given a window of recent keyframes carrying frozen preintegration spans
// from their immediate predecessors (see `ImuPreintegrationSpan`), this
// class estimates a bootstrap VIO state:
//
//   - scale (mono runs only — set to 1 on stereo / metric-depth),
//   - gravity direction in the current visual-map world frame,
//   - initial accel + gyro biases,
//   - per-keyframe world-frame velocities.
//
// The algorithm is intentionally simple (loosely following the two-stage
// "inertial-only optimization" used by ORB-SLAM3 Appendix A):
//
//   1. Closed-form gyro bias from `delta_R` vs the visually observed
//      rotation between consecutive KFs, solved as a 3×3 LSQ using the
//      first-order Jacobian J_dR_dbg ≈ -I · dt.
//   2. Linear least squares for {scale, gravity_world, v_0…v_{N-1}} from
//      the position + velocity preintegration equations, with accel_bias
//      pinned to zero for this first pass (the local BA's accel-bias
//      parameter + BiasRandomWalkError then take it from there). The
//      resulting gravity vector is rescaled to 9.81 m/s^2 after the solve.
//
// The initializer never touches the map directly; the caller (`Tracking`)
// is responsible for re-scaling KF poses + landmarks and rotating the map
// so the estimated gravity aligns with world Z-up.
class VisualInertialInitializer {
public:
    struct Result {
        // True only if the LSQ solves produced finite values and the
        // residual norms are within tolerance. Callers should fall back
        // to the pre-init behaviour otherwise.
        bool converged = false;

        // Estimated map scale (mono): multiply KF translations + landmark
        // positions by `scale` to convert to metric units. 1.0 for stereo.
        double scale = 1.0;

        // Gravity vector in the current (pre-rotation) visual world frame,
        // expressed in m/s^2. After the caller rotates the map so this
        // vector aligns with (0, 0, -9.81), downstream code can use
        // gravity_w = Vec3(0,0,-9.81) straight from the VelocityPreintegrationError.
        Vec3 gravity_w = Vec3(0.0, 0.0, -9.81);

        // Accel / gyro biases recovered from the window. accel_bias is
        // pinned to zero in the current first-pass implementation.
        Vec3 accel_bias = Vec3::Zero();
        Vec3 gyro_bias = Vec3::Zero();

        // Per-KF velocities in the (pre-rotation) visual world frame.
        // Size matches the input keyframe window.
        std::vector<Vec3> velocities;

        // Residual norms of the two LSQ stages (for logging / debugging).
        double rotation_residual_rms = 0.0;
        double linear_residual_rms = 0.0;

        // Optional human-readable message populated on failure.
        std::string message;
    };

    struct Options {
        // When true, treat the visual map as already metric-scale and
        // clamp `scale` to 1.0 in the linear stage. Set this for stereo
        // / RGB-D datasets. Mono init should leave it false.
        bool metric_scale = false;

        // Maximum absolute deviation of the estimated scale from 1.0 we
        // will accept (mono only). Helps reject degenerate windows where
        // the IMU sees no translation excitation.
        double max_scale_ratio_deviation = 10.0;

        // Gravity magnitude we expect at the surface. Used to re-normalize
        // the estimated gravity vector so downstream BA code can assume
        // |g| = 9.81.
        double gravity_magnitude = 9.81;

        // When `assume_gravity_w` is finite, the linear stage treats
        // gravity as a fixed vector (the caller supplies the accel-aligned
        // (0, 0, -9.81) world-frame vector) and solves only for
        // {scale, per-KF velocities}. This breaks the scale/gravity sign
        // ambiguity when the visual map is already gravity-aligned at
        // initialization time (the EuRoC mono path).
        //
        // Set to Vec3::Zero() to enable the full 3-DoF gravity solve.
        Vec3 assume_gravity_w = Vec3(0.0, 0.0, -9.81);

        // Soft prior pulling the estimated scale toward `scale_prior`
        // (units: map-visual per metric). Heuristic: if mono init ran a
        // 1/median_depth rescale with median ≈ 1–10 m, the recovered
        // scale should be in that range. The prior breaks the small sign
        // ambiguity that can arise when the visual world frame is
        // slightly rotated relative to the "true" gravity-aligned frame
        // and the linear LSQ finds two near-equivalent fits.
        double scale_prior = 1.0;
        double scale_prior_weight = 1.0;  // 0 disables

        // Residual norm thresholds above which we declare the solve
        // unreliable. Tightened rotation threshold (0.08 rad ≈ 4.6°)
        // rejects windows where the visual rotations diverge from the
        // IMU-integrated delta_R by more than bias alone can explain —
        // otherwise the BA rotation residual (Stage 0c.d) fights the
        // un-reconciled mismatch and the trajectory degrades. Observed
        // rot_rms: 0.13 rad on MH_01's first 15 mono KFs (rejected),
        // 0.05 rad on V1_01's (accepted).
        double rotation_residual_rms_max = 0.08;   // rad per pair
        double linear_residual_rms_max = 3.0;      // m / (m/s) per pair

        // Minimum cumulative preintegration duration across the window.
        // Short windows (≈0.5 s) leave scale/gravity poorly observable.
        double min_total_dt_seconds = 1.0;

        // Tikhonov regularization on the gyro-bias LSQ: information matrix
        // gets (1 / sigma^2) added on its diagonal. Short windows leave the
        // bias direction underdetermined, so unregularized solves can blow
        // up. Default is 0 (disabled) because with EuRoC's typical per-KF
        // dt the regularization strong enough to suppress blow-up also
        // kills the real signal; the cap below is the primary guardrail.
        // Set to a positive value to enable Tikhonov damping.
        double gyro_bias_prior_sigma = 0.0;

        // Hard cap on the estimated gyro-bias magnitude. If the
        // unregularized solve lands outside this box, the bias update is
        // discarded (falls back to zero) and the BA bias blocks pick up
        // the slack. Measured EuRoC biases are O(0.01 rad/s); the cap
        // catches the O(0.1+) over-fit we observed on MH_01 / V1_01.
        double gyro_bias_magnitude_cap = 0.05;  // rad/s
    };

    VisualInertialInitializer() = default;
    explicit VisualInertialInitializer(const Options& options) : options_(options) {}

    const Options& options() const { return options_; }

    // Run the two-stage solve over a monotonically-ordered (by time) window
    // of keyframes. Each KF (except the first) must carry a valid
    // `prev_imu_span_` whose `from_kf_id` matches the predecessor.
    Result initialize(const std::vector<Keyframe::Ptr>& keyframes) const;

    // Apply the first-order gyro-bias correction to the spans attached to
    // `keyframes`. After Result::converged, callers should invoke this so
    // the BA's rotation residual sees delta_R values consistent with the
    // newly-estimated bias. Uses the short-interval first-order Jacobian
    // J_r ≈ I and updates `span.bias_gyro` so future BA bias updates are
    // measured relative to the corrected reference.
    void applyGyroBiasCorrectionToSpans(
        const std::vector<Keyframe::Ptr>& keyframes,
        const Result& result) const;

private:
    Options options_;
};

}  // namespace svslam
