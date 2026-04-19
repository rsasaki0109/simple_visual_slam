#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>

#include "core/common.h"
#include "core/frame.h"
#include "core/map.h"
#include "core/reference_keyframe_policy.h"
#include "tracking/initializer.h"
#include "tracking/visual_inertial_initializer.h"
#include "backend/local_mapping.h"
#include "io/tum_dataset.h"
#include "sensors/imu.h"
#include <memory>
#include <mutex>

namespace svslam {

enum class TrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
};

// Aggregated counters for a tracking session (useful for health monitoring / automation).
struct TrackingRunStatistics {
    uint64_t reloc_attempts = 0;
    uint64_t reloc_successes = 0;
    uint64_t frames_tracking_lost = 0;
    uint64_t reinit_successes = 0;
};

class Tracking {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Tracking>;

    Tracking();

    void setMap(std::shared_ptr<Map> map);
    void setLocalMapping(std::shared_ptr<LocalMapping> local_mapping);
    void setReferenceKeyframePolicy(std::unique_ptr<ReferenceKeyframePolicy> policy);
    bool addFrame(Frame::Ptr frame);

    TrackingRunStatistics runStatistics() const { return run_stats_; }

    static bool shouldAcceptRecomputedPose(double err_before, double err_after);
    static bool shouldAcceptLocalMapPoseUpdate(std::size_t support,
                                               std::size_t prior_support,
                                               bool used_global_fallback,
                                               double trans_change,
                                               double rot_change,
                                               int stabilization_frames_remaining);
    static bool shouldConsiderRelocalizationCandidate(double distance_to_anchor,
                                                      bool is_reference_candidate,
                                                      bool pending_loop_correction,
                                                      int stabilization_frames_remaining);

    // Callback from LocalMapping when BA is completed
    void onBACompleted();

    // Callback from LoopClosing when pose graph correction is applied
    void onLoopCorrected();

    TrackingState state_;
    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;

    // Initialization
    Frame::Ptr initial_frame_;
    Initializer::Ptr initializer_;

    // Map reference
    std::shared_ptr<Map> map_;
    std::shared_ptr<LocalMapping> local_mapping_;

    // Motion Model
    // T_cw_current = velocity_ * T_cw_last
    SE3 velocity_;

    int num_tracked_features_ = 0;
    Keyframe::Ptr reference_keyframe_;
    Keyframe::Ptr previous_reference_keyframe_;

    // Accelerometer data for gravity alignment and stationary detection
    std::vector<AccelEntry> accel_buffer_;
    bool gravity_aligned_ = false;

    // Full IMU (accel + gyro) buffer — populated when the dataset exposes
    // imu0 (EuRoC). Stage 0b scaffolding: tracking still consults
    // accel_buffer_ for gravity / stationary detection; imu_buffer_ is held
    // here so a future VIO path (preintegration, velocity predict) can read
    // it without threading more state through the call sites.
    std::vector<ImuEntry> imu_buffer_;

    // IMU→camera extrinsic (T_cam_imu: transforms IMU/body frame points into
    // the camera frame). Defaults to identity for datasets without a known
    // extrinsic (TUM, or EuRoC without sensor.yaml). Set from EurocDataset
    // in run_mono before the first frame arrives.
    SE3 T_cam_imu_;
    bool has_cam_imu_extrinsic_ = false;

    void setImuToCameraExtrinsic(const SE3& T_cam_imu) {
        T_cam_imu_ = T_cam_imu;
        has_cam_imu_extrinsic_ = true;
    }

    // Snapshot of the VI init state for monitoring / automation.
    bool visualInertialInitCompleted() const { return vi_init_done_; }
    double visualInertialInitScale() const { return vi_init_scale_; }
    const Vec3& visualInertialInitGravity() const { return vi_init_gravity_w_; }

private:
    struct RecoveryState {
        int lost_frame_count = 0;
        int consecutive_tracking_failures = 0;
        int stabilization_frames_remaining = 0;
        SE3 last_good_pose = SE3();
    };

    struct LoopCorrectionState {
        bool pending = false;
        int pending_deferrals = 0;
        bool skip_velocity_update_once = false;
        bool force_keyframe_insertion_once = false;
        bool force_reference_refresh_once = false;
    };

    struct ReinitializationState {
        Frame::Ptr reference_frame;
        Initializer::Ptr initializer;
    };

    bool initializeWithDepth();
    void createLandmarksFromDepth(Keyframe::Ptr kf);
    bool track();
    bool initialize();
    bool trackReferenceKeyframe();
    bool trackLocalMap();
    bool needNewKeyframe();
    bool applyPendingLoopCorrection(const char* phase);
    bool recomputeCurrentPose();
    bool relocalize();  // Attempt to recover from tracking loss
    bool reinitialize();  // Re-initialize from scratch when lost for too long
    void setReferenceKeyframe(Keyframe::Ptr kf);
    void setKeyframeGravity(Keyframe::Ptr kf);  // Set gravity from accel data
    // Preintegrate imu_buffer_ between last_frame_ and current_frame_ and
    // write an IMU-predicted world-frame velocity into current_frame_.
    // No-op unless gravity_aligned_ and IMU samples span the interval.
    void predictVelocityFromImu();
    // Fold the post-tracking visual pose delta into current_frame_->velocity_
    // so Keyframe::velocity_ (and therefore the BA velocity prior) reflects a
    // visually-corrected IMU estimate instead of pure open-loop integration.
    void reconcileVelocityWithVisual();
    // Preintegrate imu_buffer_ from prev_kf's timestamp to kf's timestamp and
    // attach the resulting span to kf->prev_imu_span_ for BA consumption.
    void populateKeyframeImuSpan(const std::shared_ptr<Keyframe>& kf,
                                 const std::shared_ptr<Keyframe>& prev_kf);
    // Try to bootstrap the VIO estimate (scale, gravity, biases, velocities)
    // by running VisualInertialInitializer on the first N KFs. On success
    // the map is re-scaled + rotated so gravity aligns with world Z-up,
    // biases and velocities are written back to the KFs, and
    // vi_init_done_ flips to true so the BA preintegration residual is
    // unblocked. No-op on TUM / datasets without an IMU stream.
    void tryVisualInertialInit();
    static std::size_t countValidFrameLandmarks(const Frame::Ptr& frame);

    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::mutex pose_mutex_;  // For thread-safe pose updates

    static constexpr int max_lost_frames_ = 30;  // Max frames before giving up
    std::unique_ptr<ReferenceKeyframePolicy> reference_keyframe_policy_;
    static constexpr int max_loop_correction_deferrals_ = 6;
    static constexpr size_t min_loop_correction_correspondences_ = 80;
    static constexpr int recovery_stabilization_window_frames_ = 3;
    static constexpr double loop_relocalization_radius_m_ = 2.5;
    static constexpr double recovery_relocalization_radius_m_ = 4.0;
    static constexpr std::size_t min_stable_support_ = 120;
    static constexpr double recovery_max_change_strict_ = 0.12;
    static constexpr double recovery_max_change_relaxed_ = 0.18;
    RecoveryState recovery_state_;
    int frames_since_successful_relocalization_ = std::numeric_limits<int>::max();
    LoopCorrectionState loop_correction_state_;

    // Re-initialization state
    ReinitializationState reinitialization_state_;
    static constexpr int reinit_trigger_frames_ = 20;  // Start re-init after this many lost frames

    TrackingRunStatistics run_stats_;

    // Visual-Inertial Initialization (VIO Stage 0c.e). Once complete, the
    // map is in metric scale with gravity along world -Z, per-KF biases +
    // velocities are seeded, and downstream BA can tightly couple the
    // preintegration residual. Before completion, BA should suppress the
    // preintegration residual and fall back to the loose velocity prior.
    bool vi_init_done_ = false;
    int vi_init_attempts_ = 0;
    double vi_init_scale_ = 1.0;
    Vec3 vi_init_gravity_w_ = Vec3(0.0, 0.0, -9.81);
    static int readVioMinInitKeyframes();
};

}
