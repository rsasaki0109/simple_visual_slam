#include "tracking/tracking.h"
#include "core/heuristic_reference_keyframe_policy.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "core/keyframe.h"
#include "core/landmark.h"
#include "sensors/accelerometer.h"

namespace svslam {

namespace {

constexpr float kMaxDepthLandmarkMeters = 10.0f;
constexpr double kMinTrackedDepthMeters = 0.15;
constexpr double kMaxTrackedDepthMeters = 18.0;
constexpr double kMaxIndoorCameraPositionMeters = 50.0;
constexpr std::size_t kMinTrackLocalMapLandmarks = 250;
constexpr std::size_t kMinBootstrapCorrespondences = 30;
constexpr std::size_t kMaxBootstrapMatches = 200;
constexpr std::size_t kMinTrackLocalMapInliers = 12;
constexpr std::size_t kMinTrackReferenceInliers = 15;
constexpr std::size_t kMinPoseRecomputeCorrespondences = 10;
constexpr std::size_t kMinPoseRecomputeInliers = 20;
constexpr std::size_t kMaxDepthLandmarksPerKeyframe = 600;
// Defer low-tracked-features emergency KF insertion for N frames after a
// successful relocalization to avoid KF bursts during recovery.
constexpr int kPostRelocEmergencyKfCooldownFrames = 3;

struct PoseChange {
    double translation = std::numeric_limits<double>::infinity();
    double rotation = std::numeric_limits<double>::infinity();
};

Landmark::Ptr createDepthLandmark(const Keyframe::Ptr& kf,
                                  std::size_t keypoint_index,
                                  unsigned long& next_landmark_id) {
    if (!kf || !kf->camera_) {
        return nullptr;
    }
    if (keypoint_index >= kf->keypoints_.size() ||
        keypoint_index >= kf->landmarks_.size() ||
        kf->descriptors_.empty() ||
        keypoint_index >= static_cast<std::size_t>(kf->descriptors_.rows)) {
        return nullptr;
    }

    const auto& keypoint = kf->keypoints_[keypoint_index];
    const float depth = kf->getDepth(keypoint.pt.x, keypoint.pt.y);
    if (depth <= 0.0f || depth > kMaxDepthLandmarkMeters) {
        return nullptr;
    }

    const Vec3 p_norm = kf->camera_->unproject(Vec2(keypoint.pt.x, keypoint.pt.y));
    const Vec3 p_w = kf->T_cw_.inverse() * (p_norm * static_cast<double>(depth));

    auto landmark = std::make_shared<Landmark>(next_landmark_id++, p_w);
    landmark->addObservation(kf, keypoint_index);
    landmark->descriptor_ = kf->descriptors_.row(static_cast<int>(keypoint_index)).clone();
    return landmark;
}

SE3 poseFromOpenCvPose(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat rotation_cv;
    cv::Rodrigues(rvec, rotation_cv);

    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    cv::cv2eigen(rotation_cv, rotation);
    cv::cv2eigen(tvec, translation);
    return SE3(rotation, translation);
}

PoseChange computePoseChange(const SE3& new_pose, const SE3& reference_pose) {
    const Vec3 delta_t = new_pose.translation() - reference_pose.translation();
    const Sophus::SO3d delta_rot = new_pose.so3().inverse() * reference_pose.so3();
    const Eigen::AngleAxisd angle_axis(delta_rot.matrix());
    return {delta_t.norm(), std::abs(angle_axis.angle())};
}

bool isCameraPositionWithinBounds(const SE3& pose,
                                  double max_abs_position = kMaxIndoorCameraPositionMeters) {
    const Vec3 camera_position = pose.inverse().translation();
    return std::abs(camera_position.x()) <= max_abs_position &&
           std::abs(camera_position.y()) <= max_abs_position &&
           std::abs(camera_position.z()) <= max_abs_position;
}

template <typename KeypointIndexForCorrespondence>
bool refinePnPInliers(const Frame::Ptr& frame,
                      const std::vector<cv::Point3f>& object_points,
                      const std::vector<cv::Point2f>& image_points,
                      const std::vector<int>& inlier_indices,
                      const KeypointIndexForCorrespondence& keypoint_index_for_corr,
                      double base_gate_px,
                      cv::Mat& rvec,
                      cv::Mat& tvec,
                      std::vector<int>& refined_inlier_indices) {
    if (!frame || !frame->camera_ || inlier_indices.size() < 6) {
        return false;
    }

    std::vector<cv::Point3f> refine_object_points;
    std::vector<cv::Point2f> refine_image_points;
    std::vector<int> refine_indices;
    refine_object_points.reserve(inlier_indices.size());
    refine_image_points.reserve(inlier_indices.size());
    refine_indices.reserve(inlier_indices.size());

    for (const int index : inlier_indices) {
        if (index < 0 || index >= static_cast<int>(object_points.size())) {
            continue;
        }
        refine_object_points.push_back(object_points[index]);
        refine_image_points.push_back(image_points[index]);
        refine_indices.push_back(index);
    }
    if (refine_object_points.size() < 6) {
        return false;
    }

    cv::Mat refined_rvec = rvec.clone();
    cv::Mat refined_tvec = tvec.clone();
    bool ok = cv::solvePnP(refine_object_points, refine_image_points, frame->camera_->K(),
                           cv::Mat(), refined_rvec, refined_tvec, true,
                           cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
        return false;
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(refine_object_points, refined_rvec, refined_tvec,
                      frame->camera_->K(), cv::Mat(), projected_points);

    std::vector<cv::Point3f> gated_object_points;
    std::vector<cv::Point2f> gated_image_points;
    std::vector<int> gated_indices;
    gated_object_points.reserve(refine_object_points.size());
    gated_image_points.reserve(refine_image_points.size());
    gated_indices.reserve(refine_indices.size());

    for (std::size_t i = 0; i < projected_points.size(); ++i) {
        const int corr_index = refine_indices[i];
        const int keypoint_index = keypoint_index_for_corr(corr_index);
        const int octave =
            (keypoint_index >= 0 &&
             keypoint_index < static_cast<int>(frame->keypoints_.size()))
                ? frame->keypoints_[keypoint_index].octave
                : 0;
        const double gate_px =
            base_gate_px * (1.0 + 0.10 * static_cast<double>(std::max(0, octave)));
        const double dx = projected_points[i].x - refine_image_points[i].x;
        const double dy = projected_points[i].y - refine_image_points[i].y;
        if ((dx * dx + dy * dy) > gate_px * gate_px) {
            continue;
        }
        gated_object_points.push_back(refine_object_points[i]);
        gated_image_points.push_back(refine_image_points[i]);
        gated_indices.push_back(corr_index);
    }

    if (gated_object_points.size() < 6) {
        return false;
    }
    if (gated_object_points.size() != refine_object_points.size()) {
        refined_rvec = rvec.clone();
        refined_tvec = tvec.clone();
        ok = cv::solvePnP(gated_object_points, gated_image_points, frame->camera_->K(),
                          cv::Mat(), refined_rvec, refined_tvec, true,
                          cv::SOLVEPNP_ITERATIVE);
        if (!ok) {
            return false;
        }
    }

    rvec = refined_rvec;
    tvec = refined_tvec;
    refined_inlier_indices.swap(gated_indices);
    return true;
}

double computeAverageReprojectionError(const Frame::Ptr& frame,
                                       const SE3& pose,
                                       const std::vector<cv::Point3f>& pts3d,
                                       const std::vector<cv::Point2f>& pts2d,
                                       const std::vector<int>& indices,
                                       int* valid_count = nullptr) {
    if (!frame || !frame->camera_) {
        if (valid_count) {
            *valid_count = 0;
        }
        return std::numeric_limits<double>::infinity();
    }

    double error_sum = 0.0;
    int count = 0;
    for (const int index : indices) {
        if (index < 0 || index >= static_cast<int>(pts3d.size())) {
            continue;
        }

        const auto& point = pts3d[index];
        const auto& observation = pts2d[index];
        const Vec3 point_camera = pose * Vec3(point.x, point.y, point.z);
        if (point_camera.z() <= 0.0 || !point_camera.allFinite()) {
            continue;
        }

        const Vec2 projection = frame->camera_->project(point_camera);
        const double dx = observation.x - projection[0];
        const double dy = observation.y - projection[1];
        error_sum += std::sqrt(dx * dx + dy * dy);
        ++count;
    }

    if (valid_count) {
        *valid_count = count;
    }
    if (count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return error_sum / static_cast<double>(count);
}

void assignFrameLandmarksFromInliers(const Frame::Ptr& frame,
                                     const std::vector<int>& keypoint_indices,
                                     const std::vector<Landmark::Ptr>& landmarks,
                                     const std::vector<int>& inliers) {
    if (!frame) {
        return;
    }

    // Hold frame->mutex_ around the assign + writes: onBACompleted may be
    // snapshotting frame->landmarks_ on the LocalMapping thread at the same
    // time.
    std::lock_guard<std::mutex> lock(frame->mutex_);
    frame->landmarks_.assign(frame->keypoints_.size(), nullptr);
    for (const int index : inliers) {
        if (index < 0 || index >= static_cast<int>(keypoint_indices.size()) ||
            index >= static_cast<int>(landmarks.size())) {
            continue;
        }
        const int keypoint_index = keypoint_indices[index];
        if (keypoint_index < 0 ||
            keypoint_index >= static_cast<int>(frame->landmarks_.size())) {
            continue;
        }
        frame->landmarks_[keypoint_index] = landmarks[index];
    }
}

}  // namespace

Tracking::Tracking() : state_(TrackingState::NO_IMAGES_YET) {
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    reference_keyframe_policy_ = std::make_unique<HeuristicReferenceKeyframePolicy>();
}

void Tracking::setMap(std::shared_ptr<Map> map) {
    map_ = map;
}

void Tracking::setLocalMapping(std::shared_ptr<LocalMapping> local_mapping) {
    local_mapping_ = local_mapping;
}

void Tracking::setReferenceKeyframePolicy(std::unique_ptr<ReferenceKeyframePolicy> policy) {
    if (!policy) return;
    reference_keyframe_policy_ = std::move(policy);
}

void Tracking::setReferenceKeyframe(Keyframe::Ptr kf) {
    if (reference_keyframe_ == kf) return;
    previous_reference_keyframe_ = reference_keyframe_;
    reference_keyframe_ = kf;
}

bool Tracking::addFrame(Frame::Ptr frame) {
    // Only the main thread writes current_frame_ / last_frame_, but the
    // LocalMapping on_ba_completed_ callback reads current_frame_ under
    // pose_mutex_. Without the matching lock on this side, TSan flags the
    // shared_ptr swap as a data race. Hold pose_mutex_ only across the
    // swap itself so the rest of addFrame -- which is heavy and internally
    // synchronizes via Frame / Keyframe mutexes -- does not block the
    // LocalMapping thread.
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        current_frame_ = frame;
    }

    if (state_ == TrackingState::NO_IMAGES_YET) {
        state_ = TrackingState::NOT_INITIALIZED;
    }

    bool success = false;
    if (state_ == TrackingState::NOT_INITIALIZED) {
        success = initialize();
    } else {
        success = track();
    }

    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        last_frame_ = current_frame_;
    }
    return success;
}

bool Tracking::initializeWithDepth() {
    current_frame_->setPose(SE3());

    if (!gravity_aligned_ && !accel_buffer_.empty()) {
        Vec3 gravity = AccelerometerProcessor::estimateGravity(accel_buffer_);
        if (gravity.norm() > 0.5) {
            Mat33 R_align = AccelerometerProcessor::computeGravityAlignment(gravity);
            SE3 T_aligned(R_align, Vec3(0, 0, 0));
            current_frame_->setPose(T_aligned);
            gravity_aligned_ = true;
            std::cout << "Tracking: Applied gravity alignment" << std::endl;
        }
    }

    auto kf = std::make_shared<Keyframe>(current_frame_);
    setKeyframeGravity(kf);

    int created = 0;
    static unsigned long depth_lm_id = 200000;

    // kf is not yet published to the map, so kf->landmarks_ writes are
    // single-threaded. current_frame_ is published; hold its mutex_ across
    // the loop to avoid racing with onBACompleted reads.
    std::lock_guard<std::mutex> frame_lock(current_frame_->mutex_);
    for (size_t i = 0; i < kf->keypoints_.size(); ++i) {
        auto lm = createDepthLandmark(kf, i, depth_lm_id);
        if (!lm) {
            continue;
        }
        kf->landmarks_[i] = lm;
        current_frame_->landmarks_[i] = lm;

        if (map_) map_->addLandmark(lm);
        created++;
    }

    if (created < 100) {
        std::cout << "Tracking: Depth init failed - only " << created << " points with valid depth" << std::endl;
        // Clean up
        if (map_) {
            for (auto& lm_pair : map_->getAllLandmarks()) {
                map_->removeLandmark(lm_pair.second);
            }
        }
        return false;
    }

    if (map_) {
        map_->addKeyframe(kf);
    }

    setReferenceKeyframe(kf);
    initial_frame_ = current_frame_;
    state_ = TrackingState::OK;

    std::cout << "Tracking: Depth-based initialization SUCCESS! " << created
              << " 3D points from single frame (metric scale)" << std::endl;
    return true;
}

void Tracking::createLandmarksFromDepth(Keyframe::Ptr kf) {
    if (!kf || kf->depth_image_.empty()) return;

    int created = 0;
    static unsigned long depth_track_lm_id = 300000;

    struct DepthCandidate {
        std::size_t keypoint_index = 0;
        float depth = 0.0f;
    };
    std::vector<DepthCandidate> depth_candidates;
    depth_candidates.reserve(kf->keypoints_.size());

    for (size_t i = 0; i < kf->keypoints_.size(); ++i) {
        if (kf->landmarks_[i]) continue;
        const auto& keypoint = kf->keypoints_[i];
        const float depth = kf->getDepth(keypoint.pt.x, keypoint.pt.y);
        if (depth <= 0.0f || depth > kMaxDepthLandmarkMeters) {
            continue;
        }
        depth_candidates.push_back({i, depth});
    }

    std::sort(depth_candidates.begin(), depth_candidates.end(),
              [&](const DepthCandidate& lhs, const DepthCandidate& rhs) {
                  const auto& kp_lhs = kf->keypoints_[lhs.keypoint_index];
                  const auto& kp_rhs = kf->keypoints_[rhs.keypoint_index];
                  if (kp_lhs.octave != kp_rhs.octave) {
                      return kp_lhs.octave < kp_rhs.octave;
                  }
                  if (kp_lhs.response != kp_rhs.response) {
                      return kp_lhs.response > kp_rhs.response;
                  }
                  if (lhs.depth != rhs.depth) {
                      return lhs.depth < rhs.depth;
                  }
                  return lhs.keypoint_index < rhs.keypoint_index;
              });

    const std::size_t max_candidates =
        std::min(kMaxDepthLandmarksPerKeyframe, depth_candidates.size());
    for (std::size_t candidate_idx = 0; candidate_idx < max_candidates; ++candidate_idx) {
        const std::size_t i = depth_candidates[candidate_idx].keypoint_index;
        auto lm = createDepthLandmark(kf, i, depth_track_lm_id);
        if (!lm) {
            continue;
        }
        kf->landmarks_[i] = lm;
        if (map_) map_->addLandmark(lm);
        created++;
    }

    if (created > 0) {
        std::cout << "Tracking: Created " << created << " landmarks from depth in KF " << kf->id_ << std::endl;
    }
}

bool Tracking::initialize() {
    if (!initial_frame_) {
        // First frame - try single-frame depth initialization
        if (!current_frame_->depth_image_.empty()) {
            initial_frame_ = current_frame_;
            if (initializeWithDepth()) {
                return true;
            }
            // Depth init failed, fall through to two-frame init
            initial_frame_ = nullptr;
        }

        // First frame
        initial_frame_ = current_frame_;
        // Set identity pose
        initial_frame_->setPose(SE3());

        // Initialize initializer
        initializer_ = std::make_shared<Initializer>(initial_frame_);

        std::cout << "Tracking: Initial Frame set (ID: " << initial_frame_->id_ << ")" << std::endl;
        return true;
    } else {
        // Second frame, try to initialize
        if (initializer_->initialize(current_frame_)) {
            std::cout << "Tracking: Initialization SUCCESS!" << std::endl;

            // Gravity alignment (mono init): rotate the world frame so gravity
            // points in [0, 0, -1], matching the assumption in the BA gravity
            // prior. Without this, GravityPriorError silently no-ops on mono
            // because gravity_aligned_ stays false. The initializer operates in
            // c1-frame coordinates and is independent of world, so we can apply
            // the alignment here and then rewrite all poses and landmarks.
            SE3 T_align;  // identity by default (no-op when no accel)
            if (!gravity_aligned_ && !accel_buffer_.empty()) {
                Vec3 gravity = AccelerometerProcessor::estimateGravity(accel_buffer_);
                if (gravity.norm() > 0.5) {
                    Mat33 R_align = AccelerometerProcessor::computeGravityAlignment(gravity);
                    T_align = SE3(R_align, Vec3(0, 0, 0));
                    gravity_aligned_ = true;
                    std::cout << "Tracking: Applied gravity alignment (mono init)" << std::endl;
                }
            }
            initial_frame_->setPose(T_align);

            // 1. Create Keyframes (poses will be refreshed below; gravity is set
            //    now that gravity_aligned_ may have flipped to true)
            auto kf_init = std::make_shared<Keyframe>(initial_frame_);
            auto kf_cur = std::make_shared<Keyframe>(current_frame_);
            setKeyframeGravity(kf_init);
            setKeyframeGravity(kf_cur);

            // Set Pose for current (T_cw):
            // Initializer returns T_c1_c2 which we defined as T_c2_c1 (Pose of c2 w.r.t c1).
            // With initial at T_align, T_c2_w_new = T_c2_c1 * T_c1_w_new = T_c2_c1 * T_align.
            current_frame_->setPose(initializer_->T_c1_c2_ * T_align);
            kf_cur->T_cw_ = current_frame_->getPose();
            kf_init->T_cw_ = initial_frame_->getPose();
            
            std::cout << "Tracking: Initialized Pose T_c2_c1: \n" << current_frame_->getPose().matrix() << std::endl;
            
            // 2. Create MapPoints
            size_t tri_true = 0;
            size_t inserted = 0;
            size_t rejected_nonfinite = 0;
            size_t rejected_nonpositive_z = 0;
            size_t rejected_absmax = 0;

            double z_min = std::numeric_limits<double>::infinity();
            double z_max = -std::numeric_limits<double>::infinity();
            double norm_max = 0.0;

            for (size_t i = 0; i < initializer_->is_triangulated_.size(); ++i) {
                if (initializer_->is_triangulated_[i]) {
                    tri_true++;
                    // Create Landmark
                    cv::Point3f pt3d = initializer_->triangulated_points_[i]; // In Ref frame
                    if (!std::isfinite(pt3d.x) || !std::isfinite(pt3d.y) || !std::isfinite(pt3d.z)) {
                        rejected_nonfinite++;
                        continue;
                    }
                    if (pt3d.z <= 0.0f) {
                        rejected_nonpositive_z++;
                        continue;
                    }
                    const float abs_max = 1e4f;
                    if (std::abs(pt3d.x) > abs_max || std::abs(pt3d.y) > abs_max || std::abs(pt3d.z) > abs_max) {
                        rejected_absmax++;
                        continue;
                    }

                    z_min = std::min(z_min, static_cast<double>(pt3d.z));
                    z_max = std::max(z_max, static_cast<double>(pt3d.z));
                    const double nrm = std::sqrt(static_cast<double>(pt3d.x) * pt3d.x + static_cast<double>(pt3d.y) * pt3d.y + static_cast<double>(pt3d.z) * pt3d.z);
                    norm_max = std::max(norm_max, nrm);

                    // Triangulated points come out of the initializer in c1-frame
                    // coordinates. Transform into the (possibly gravity-aligned)
                    // world frame: p_world = T_c1_w^{-1} * p_c1 = T_align^{-1} * p_c1.
                    Vec3 pos_w = T_align.inverse() * Vec3(pt3d.x, pt3d.y, pt3d.z);
                    
                    auto lm = std::make_shared<Landmark>(i, pos_w); // ID? Use global ID counter later
                    
                    // Add observations
                    // We need to know feature index in keyframe
                    int idx_ref = initializer_->matches_[i].queryIdx;
                    int idx_cur = initializer_->matches_[i].trainIdx;
                    
                    lm->addObservation(kf_init, idx_ref);
                    lm->addObservation(kf_cur, idx_cur);
                    
                    // Set descriptor (using reference frame)
                    lm->descriptor_ = initial_frame_->descriptors_.row(idx_ref).clone();

                    // Add landmarks to keyframes
                    kf_init->landmarks_[idx_ref] = lm;
                    kf_cur->landmarks_[idx_cur] = lm;

                    // Update Frames as well so they are tracked. Both frames
                    // are published (initial_frame_ may still be the target
                    // of an onBACompleted snapshot path), so take each
                    // mutex_ around the single-slot write.
                    {
                        std::lock_guard<std::mutex> lock(initial_frame_->mutex_);
                        initial_frame_->landmarks_[idx_ref] = lm;
                    }
                    {
                        std::lock_guard<std::mutex> lock(current_frame_->mutex_);
                        current_frame_->landmarks_[idx_cur] = lm;
                    }
                    
                    // Add to map
                    if (map_) {
                        map_->addLandmark(lm);
                        inserted++;
                    }
                }
            }

            std::cout << "Tracking: Init triangulation tri_true=" << tri_true
                      << " inserted=" << inserted
                      << " rejected_nonfinite=" << rejected_nonfinite
                      << " rejected_nonpositive_z=" << rejected_nonpositive_z
                      << " rejected_absmax=" << rejected_absmax
                      << " z_min=" << z_min
                      << " z_max=" << z_max
                      << " norm_max=" << norm_max
                      << std::endl;

            // Scale normalization: skip if we have metric depth data
            bool has_metric_depth = !current_frame_->depth_image_.empty() && current_frame_->depth_is_metric_;
            if (!has_metric_depth) {
                // Collect all depths for computing median
                std::vector<double> depths;
                for (auto& kv : map_->getAllLandmarks()) {
                    auto& lm = kv.second;
                    if (lm && !lm->isBad()) {
                        depths.push_back(lm->getPos().z());
                    }
                }
                if (!depths.empty()) {
                    std::sort(depths.begin(), depths.end());
                    double median_depth = depths[depths.size() / 2];
                    if (median_depth > 0.0) {
                        double scale = 1.0 / median_depth;
                        std::cout << "Tracking: Scaling map by " << scale << " (median depth was " << median_depth << ")" << std::endl;

                        for (auto& kv : map_->getAllLandmarks()) {
                            auto& lm = kv.second;
                            if (lm && !lm->isBad()) {
                                lm->setPos(lm->getPos() * scale);
                            }
                        }

                        SE3 T_scaled = SE3(kf_cur->T_cw_.so3(), kf_cur->T_cw_.translation() * scale);
                        kf_cur->T_cw_ = T_scaled;
                        current_frame_->setPose(T_scaled);
                    }
                }
            } else {
                std::cout << "Tracking: Metric depth available, skipping scale normalization" << std::endl;
            }

            if (map_) {
                map_->addKeyframe(kf_init);
                map_->addKeyframe(kf_cur);

                std::cout << "Tracking: Map after init: keyframes=" << map_->getAllKeyframes().size()
                          << " landmarks=" << map_->getAllLandmarks().size() << std::endl;
            }

            setReferenceKeyframe(kf_cur);
            state_ = TrackingState::OK;
            return true;
        } else {
            std::cout << "Tracking: Initialization failed. Retrying..." << std::endl;
            // Reset? Or just keep trying with new current vs old initial?
            // ORB-SLAM replaces initial if not enough disparity
            return false;
        }
    }
}

bool Tracking::track() {
    // 1. Motion Model Prediction
    if (last_frame_) {
        // Check accelerometer for stationary detection
        bool stationary = false;
        if (!accel_buffer_.empty() && last_frame_) {
            auto recent_accel = std::vector<AccelEntry>();
            for (const auto& a : accel_buffer_) {
                if (a.timestamp_sec >= last_frame_->timestamp_ &&
                    a.timestamp_sec <= current_frame_->timestamp_) {
                    recent_accel.push_back(a);
                }
            }
            if (AccelerometerProcessor::isStationary(recent_accel)) {
                stationary = true;
            }
        }

        if (stationary) {
            // Zero motion prediction
            current_frame_->setPose(last_frame_->getPose());
        } else {
            // T_current = velocity * T_last
            SE3 T_cw_pred = velocity_ * last_frame_->getPose();
            current_frame_->setPose(T_cw_pred);
        }
    }

    // 2. Track Reference Keyframe (Frame-to-Frame matching for now)
    bool pending_loop_applied = false;
    bool ref_tracking_ok = trackReferenceKeyframe();
    if (ref_tracking_ok) {
        pending_loop_applied = applyPendingLoopCorrection("after reference tracking");
    }

    // 3. Track Local Map
    bool local_map_ok = false;
    if (ref_tracking_ok) {
        local_map_ok = trackLocalMap();
        pending_loop_applied =
            applyPendingLoopCorrection("after local map") || pending_loop_applied;
    }

    const bool recovered_with_pending_loop =
        ref_tracking_ok && pending_loop_applied && !local_map_ok;

    // 4. Handle tracking success/failure
    if (local_map_ok || recovered_with_pending_loop) {
        if (recovered_with_pending_loop) {
            num_tracked_features_ =
                static_cast<int>(countValidFrameLandmarks(current_frame_));
            std::cout << "Tracking: Using reference-tracking pose after pending loop correction"
                      << " (tracked=" << num_tracked_features_ << ")" << std::endl;
        }
        state_ = TrackingState::OK;
        recovery_state_.consecutive_tracking_failures = 0;
        recovery_state_.lost_frame_count = 0;
        recovery_state_.last_good_pose = current_frame_->getPose();
        reinitialization_state_.reference_frame.reset();
        reinitialization_state_.initializer.reset();

        if (loop_correction_state_.skip_velocity_update_once) {
            std::cout << "Tracking: Preserving identity velocity after loop-correction handoff" << std::endl;
            loop_correction_state_.skip_velocity_update_once = false;
        } else if (last_frame_) {
            velocity_ = current_frame_->getPose() * last_frame_->getPose().inverse();
        }
    } else {
        std::cout << "Tracking: Lost, attempting relocalization..." << std::endl;
        run_stats_.reloc_attempts++;

        if (relocalize()) {
            std::cout << "Tracking: Relocalization successful!" << std::endl;
            applyPendingLoopCorrection("after relocalization");
            run_stats_.reloc_successes++;
            state_ = TrackingState::OK;
            recovery_state_.consecutive_tracking_failures = 0;
            recovery_state_.lost_frame_count = 0;
            velocity_ = SE3();
            loop_correction_state_.skip_velocity_update_once = false;
            recovery_state_.stabilization_frames_remaining =
                recovery_stabilization_window_frames_;
            frames_since_successful_relocalization_ = 0;
            recovery_state_.last_good_pose = current_frame_->getPose();
            reinitialization_state_.reference_frame.reset();
            reinitialization_state_.initializer.reset();
        } else {
            state_ = TrackingState::LOST;
            ++recovery_state_.consecutive_tracking_failures;
            ++recovery_state_.lost_frame_count;

            if (recovery_state_.consecutive_tracking_failures >= 3) {
                velocity_ = SE3();
            }

            if (recovery_state_.lost_frame_count >= reinit_trigger_frames_) {
                std::cout << "Tracking: Lost for " << recovery_state_.lost_frame_count
                          << " frames, attempting re-initialization..." << std::endl;
                if (reinitialize()) {
                    std::cout << "Tracking: Re-initialization successful!" << std::endl;
                    run_stats_.reinit_successes++;
                    state_ = TrackingState::OK;
                    recovery_state_.consecutive_tracking_failures = 0;
                    recovery_state_.lost_frame_count = 0;
                    velocity_ = SE3();
                    recovery_state_.stabilization_frames_remaining =
                        recovery_stabilization_window_frames_;
                    recovery_state_.last_good_pose = current_frame_->getPose();
                }
            }
        }
        
        if (recovery_state_.lost_frame_count > max_lost_frames_) {
            std::cout << "Tracking: Completely lost for "
                      << recovery_state_.lost_frame_count << " frames" << std::endl;
        }

        loop_correction_state_.skip_velocity_update_once = false;
    }

    if (state_ == TrackingState::LOST) {
        run_stats_.frames_tracking_lost++;
    }

    if (state_ == TrackingState::OK &&
        recovery_state_.stabilization_frames_remaining > 0) {
        --recovery_state_.stabilization_frames_remaining;
    }
    if (frames_since_successful_relocalization_ < std::numeric_limits<int>::max()) {
        ++frames_since_successful_relocalization_;
    }

    // 5. Check if we need a new Keyframe (only if tracking is OK)
    if (state_ == TrackingState::OK && needNewKeyframe()) {
        std::cout << "Tracking: Insert New Keyframe (ID=" << current_frame_->id_ << ")" << std::endl;
        // Create new Keyframe
        auto kf = std::make_shared<Keyframe>(current_frame_);
        setKeyframeGravity(kf);
        for (size_t i = 0; i < kf->landmarks_.size(); ++i) {
            auto& lm = kf->landmarks_[i];
            if (!lm || lm->isBad()) continue;
            lm->addObservation(kf, i);
        }

        // Create additional landmarks from depth for unmatched keypoints
        createLandmarksFromDepth(kf);

        int candidate_landmarks = 0;
        for (const auto& lm : kf->landmarks_) {
            if (lm && !lm->isBad()) {
                candidate_landmarks++;
            }
        }

        ReferenceKeyframePolicyInput policy_input;
        policy_input.tracked_features = num_tracked_features_;
        policy_input.detected_keypoints = static_cast<int>(current_frame_->keypoints_.size());
        policy_input.candidate_landmarks = candidate_landmarks;
        policy_input.frames_since_reference = reference_keyframe_
            ? static_cast<int>(current_frame_->id_ - reference_keyframe_->id_)
            : 0;
        policy_input.lost_frames = recovery_state_.lost_frame_count;
        policy_input.has_depth = !current_frame_->depth_image_.empty();
        policy_input.has_accel = !accel_buffer_.empty();

        const auto decision = reference_keyframe_policy_
            ? reference_keyframe_policy_->evaluate(policy_input)
            : ReferenceKeyframeDecision{
                  ReferenceKeyframeAction::PromoteNewReference,
                  0.50,
                  "missing policy fallback"
              };

        if (loop_correction_state_.force_reference_refresh_once) {
            setReferenceKeyframe(kf);
            previous_reference_keyframe_.reset();
            loop_correction_state_.force_reference_refresh_once = false;
            std::cout << "Tracking: Forced reference refresh after pending loop correction expiry" << std::endl;
        } else if (decision.promoteNewReference()) {
            setReferenceKeyframe(kf);
        } else {
            std::cout << "Tracking: Keeping previous reference KF due to policy veto"
                      << " (" << decision.reason
                      << ", tracked=" << num_tracked_features_
                      << ", keypoints=" << current_frame_->keypoints_.size()
                      << ", landmarks=" << candidate_landmarks
                      << ", confidence=" << decision.confidence << ")"
                      << std::endl;
        }

        if (local_mapping_) {
            local_mapping_->insertKeyframe(kf);
        } else {
            map_->addKeyframe(kf);
        }

        loop_correction_state_.force_keyframe_insertion_once = false;
    }

    return state_ == TrackingState::OK;
}

bool Tracking::needNewKeyframe() {
    if (!map_) return false;
    if (!reference_keyframe_) return false;

    const int frames_since_reference =
        static_cast<int>(current_frame_->id_ - reference_keyframe_->id_);

    // Snapshot reference_keyframe_->landmarks_ under kf->mutex_ —
    // LocalMapping::createNewMapPoints writes the same container.
    std::vector<Landmark::Ptr> ref_landmarks_snapshot;
    {
        std::lock_guard<std::mutex> lock(reference_keyframe_->mutex_);
        ref_landmarks_snapshot = reference_keyframe_->landmarks_;
    }
    int ref_landmarks = 0;
    for (auto& lm : ref_landmarks_snapshot) {
        if (lm && !lm->isBad()) {
            ref_landmarks++;
        }
    }
    const double tracked_ratio = ref_landmarks > 0
        ? static_cast<double>(num_tracked_features_) / ref_landmarks
        : -1.0;

    if (loop_correction_state_.force_keyframe_insertion_once) {
        std::cout << "needNewKeyframe: Forced insertion after pending loop correction expiry." << std::endl;
        return true;
    }

    if (loop_correction_state_.pending) {
        std::cout << "needNewKeyframe: Deferring insertion until pending loop correction is resolved." << std::endl;
        return false;
    }

    // Heuristics for new keyframe decision:
    // 1. Min frames since last KF (mono: 4 to reduce KF bursts at the minimum gap; room logs:
    //    101->104->107 every 3 frames under stress); RGB-D keeps 3.
    const int min_frames_since_last_kf =
        current_frame_->depth_image_.empty() ? 4 : 3;
    if (frames_since_reference < min_frames_since_last_kf) {
        return false;
    }

    // 2. Track quality: if tracked features drop below threshold, insert KF
    const int min_tracked_threshold = 60;
    if (num_tracked_features_ < min_tracked_threshold) {
        if (frames_since_successful_relocalization_ <= kPostRelocEmergencyKfCooldownFrames) {
            std::cout << "needNewKeyframe: Low tracked features (" << num_tracked_features_
                      << ") within post-reloc cooldown ("
                      << frames_since_successful_relocalization_ << "/"
                      << kPostRelocEmergencyKfCooldownFrames << "), deferring." << std::endl;
            return false;
        }
        std::cout << "needNewKeyframe: Low tracked features (" << num_tracked_features_ << "), inserting KF." << std::endl;
        return true;
    }

    // 3. Max frames since last KF
    const int max_frames_since_last_kf = 12;
    if (frames_since_reference >= max_frames_since_last_kf) {
        std::cout << "needNewKeyframe: Max frames reached, inserting KF." << std::endl;
        return true;
    }

    // 4. Ratio of tracked vs reference KF landmarks
    if (ref_landmarks > 0) {
        const bool mono_sparse_reference =
            current_frame_->depth_image_.empty() &&
            current_frame_->id_ >= 50 &&
            current_frame_->id_ < 80 &&
            ref_landmarks < 1300 &&
            num_tracked_features_ < 900;
        const double tracked_ratio_threshold = mono_sparse_reference ? 0.70 : 0.65;
        if (tracked_ratio < tracked_ratio_threshold) {
            std::cout << "needNewKeyframe: Low tracking ratio (" << tracked_ratio << "), inserting KF." << std::endl;
            return true;
        }
        return false;
    }

    return false;
}

bool Tracking::trackLocalMap() {
    if (!map_) return false;

    // Skip if loop closing is correcting the map to avoid accessing invalidated landmarks
    if (map_->loop_correcting_.load()) {
        std::cout << "TrackLocalMap: Loop correction in progress, skipping" << std::endl;
        return true;
    }

    // 1. Build a true local map from the reference keyframe and its covisible neighbors.
    enum class LocalMapSourceBucket : std::size_t {
        Reference = 0,
        ReferenceNeighbor,
        PreviousReference,
        PreviousReferenceNeighbor,
        GlobalFallback,
        Count
    };
    constexpr std::size_t kNumLocalMapSourceBuckets =
        static_cast<std::size_t>(LocalMapSourceBucket::Count);
    const auto source_bucket_name = [](LocalMapSourceBucket bucket) {
        switch (bucket) {
            case LocalMapSourceBucket::Reference: return "ref";
            case LocalMapSourceBucket::ReferenceNeighbor: return "ref_neighbors";
            case LocalMapSourceBucket::PreviousReference: return "prev_ref";
            case LocalMapSourceBucket::PreviousReferenceNeighbor: return "prev_neighbors";
            case LocalMapSourceBucket::GlobalFallback: return "global";
            case LocalMapSourceBucket::Count: break;
        }
        return "unknown";
    };

    struct LocalMapSourceStats {
        std::array<std::size_t, kNumLocalMapSourceBuckets> pool_added{};
    };

    std::vector<Landmark::Ptr> landmarks;
    std::vector<LocalMapSourceBucket> landmark_source_buckets;
    std::set<unsigned long> landmark_ids;
    LocalMapSourceStats local_map_source_stats;

    auto add_landmarks_from_kf = [&](const Keyframe::Ptr& kf,
                                     LocalMapSourceBucket source_bucket) {
        if (!kf) return;
        // Snapshot under kf->mutex_ so we don't race with LocalMapping
        // writing kf->landmarks_[i] = lm in createNewMapPoints. TSan flagged
        // shared_ptr<Landmark>::operator bool() reads here tearing against
        // the concurrent shared_ptr assignment on the mapping thread.
        std::vector<Landmark::Ptr> snapshot;
        {
            std::lock_guard<std::mutex> lock(kf->mutex_);
            snapshot = kf->landmarks_;
        }
        for (const auto& lm : snapshot) {
            if (!lm || lm->isBad()) continue;
            if (!landmark_ids.insert(lm->id_).second) continue;
            landmarks.push_back(lm);
            landmark_source_buckets.push_back(source_bucket);
            ++local_map_source_stats.pool_added[static_cast<std::size_t>(source_bucket)];
        }
    };

    add_landmarks_from_kf(reference_keyframe_, LocalMapSourceBucket::Reference);
    if (reference_keyframe_) {
        const auto reference_neighbors = reference_keyframe_->getBestCovisibilityKeyframes(15);
        for (const auto& neighbor : reference_neighbors) {
            add_landmarks_from_kf(neighbor, LocalMapSourceBucket::ReferenceNeighbor);
        }
    }

    if (landmarks.size() < kMinTrackLocalMapLandmarks &&
        previous_reference_keyframe_ &&
        previous_reference_keyframe_ != reference_keyframe_) {
        add_landmarks_from_kf(previous_reference_keyframe_, LocalMapSourceBucket::PreviousReference);
        const auto previous_neighbors = previous_reference_keyframe_->getBestCovisibilityKeyframes(10);
        for (const auto& neighbor : previous_neighbors) {
            add_landmarks_from_kf(neighbor, LocalMapSourceBucket::PreviousReferenceNeighbor);
        }
        std::cout << "TrackLocalMap: Augmented with previous reference KF "
                  << previous_reference_keyframe_->id_ << std::endl;
    }

    if (landmarks.size() < kMinTrackLocalMapLandmarks) {
        const auto& all_landmarks = map_->getAllLandmarks();
        for (const auto& kv : all_landmarks) {
            const auto& lm = kv.second;
            if (!lm || lm->isBad()) continue;
            if (!landmark_ids.insert(lm->id_).second) continue;
            landmarks.push_back(lm);
            landmark_source_buckets.push_back(LocalMapSourceBucket::GlobalFallback);
            ++local_map_source_stats.pool_added[static_cast<std::size_t>(LocalMapSourceBucket::GlobalFallback)];
        }
        std::cout << "TrackLocalMap: Expanded to global map fallback with "
                  << landmarks.size() << " landmarks" << std::endl;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<std::shared_ptr<Landmark>> matched_landmarks; // Keep track of LM for each point
    std::vector<int> matched_kp_indices; // Keep track of KP index for each point

    std::vector<bool> keypoint_already_matched(current_frame_->keypoints_.size(), false);

    bool used_global_fallback = false;
    std::size_t prior_support = 0;
    for (const auto& lm : current_frame_->landmarks_) {
        if (!lm || lm->isBad()) continue;
        const Vec3 pos = lm->getPos();
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) continue;
        ++prior_support;
    }

    // For fallback matching, only consider landmarks that are in the current view frustum
    cv::Mat visible_lm_descs;
    std::vector<Landmark::Ptr> visible_lm_list;
    std::vector<cv::Point3f> visible_lm_pts;
    std::vector<LocalMapSourceBucket> visible_lm_source_buckets;
    
    std::cout << "TrackLocalMap: Landmarks to check: " << landmarks.size() << std::endl;
    std::cout << "TrackLocalMap: LocalMapSources";
    for (std::size_t bucket_idx = 0; bucket_idx < kNumLocalMapSourceBuckets; ++bucket_idx) {
        const auto bucket = static_cast<LocalMapSourceBucket>(bucket_idx);
        std::cout << ' ' << source_bucket_name(bucket)
                  << '=' << local_map_source_stats.pool_added[bucket_idx];
    }
    std::cout << std::endl;

    struct PoseFilterStats {
        std::size_t focus_reject_reprojection = 0;
    };

    auto filter_correspondences_by_pose = [&](double base_gate_px,
                                              std::size_t focus_begin =
                                                  std::numeric_limits<std::size_t>::max()) {
        PoseFilterStats stats;
        if (object_points.empty()) return stats;

        std::vector<cv::Point3f> filtered_object_points;
        std::vector<cv::Point2f> filtered_image_points;
        std::vector<Landmark::Ptr> filtered_landmarks;
        std::vector<int> filtered_kp_indices;

        filtered_object_points.reserve(object_points.size());
        filtered_image_points.reserve(image_points.size());
        filtered_landmarks.reserve(matched_landmarks.size());
        filtered_kp_indices.reserve(matched_kp_indices.size());

        const SE3 T_cw_est = current_frame_->getPose();
        for (size_t i = 0; i < object_points.size(); ++i) {
            const bool in_focus = i >= focus_begin;

            const auto& Pw = object_points[i];
            Vec3 p_w(Pw.x, Pw.y, Pw.z);
            Vec3 p_c = T_cw_est * p_w;
            if (!std::isfinite(p_c.x()) || !std::isfinite(p_c.y()) || !std::isfinite(p_c.z())) {
                continue;
            }
            if (p_c[2] <= kMinTrackedDepthMeters || p_c[2] > kMaxTrackedDepthMeters) {
                continue;
            }

            const int kp_idx = matched_kp_indices[i];
            const int octave = (kp_idx >= 0 && kp_idx < static_cast<int>(current_frame_->keypoints_.size()))
                ? current_frame_->keypoints_[kp_idx].octave
                : 0;
            const double gate_px = base_gate_px * (1.0 + 0.12 * static_cast<double>(std::max(0, octave)));

            Vec2 proj = current_frame_->camera_->project(p_c);
            const auto& uv = image_points[i];
            const double dx = uv.x - proj[0];
            const double dy = uv.y - proj[1];
            if ((dx * dx + dy * dy) > gate_px * gate_px) {
                if (in_focus) {
                    ++stats.focus_reject_reprojection;
                }
                continue;
            }

            filtered_object_points.push_back(Pw);
            filtered_image_points.push_back(uv);
            filtered_landmarks.push_back(matched_landmarks[i]);
            filtered_kp_indices.push_back(kp_idx);
        }

        object_points.swap(filtered_object_points);
        image_points.swap(filtered_image_points);
        matched_landmarks.swap(filtered_landmarks);
        matched_kp_indices.swap(filtered_kp_indices);
        return stats;
    };

    // 2. Project and Match
    int matches_found = 0;
    int visible_points = 0;
    int skipped_nonfinite = 0;
    int skipped_behind_or_close = 0;
    int skipped_oob = 0;
    for (std::size_t lm_idx = 0; lm_idx < landmarks.size(); ++lm_idx) {
        const auto& lm = landmarks[lm_idx];
        const std::size_t source_idx = lm_idx < landmark_source_buckets.size()
            ? static_cast<std::size_t>(landmark_source_buckets[lm_idx])
            : static_cast<std::size_t>(LocalMapSourceBucket::GlobalFallback);
        if (!lm) continue;
        if (lm->descriptor_.empty()) continue;
        
        // Project
        Vec3 pos_w = lm->getPos();
        if (!std::isfinite(pos_w[0]) || !std::isfinite(pos_w[1]) || !std::isfinite(pos_w[2])) {
            skipped_nonfinite++;
            continue;
        }
        Vec3 pos_c = current_frame_->getPose() * pos_w; // T_cw * pos_w

        // Check depth: must be positive and within reasonable range
        const double max_depth = 20.0;  // Reasonable for indoor scenes
        if (pos_c[2] <= 0.1 || pos_c[2] > max_depth) {
            skipped_behind_or_close++;
            continue; // Behind camera, too close, or too far
        }

        Vec2 px = current_frame_->camera_->project(pos_c);

        // Check bounds
        if (px[0] < 0 || px[0] >= current_frame_->image_.cols ||
            px[1] < 0 || px[1] >= current_frame_->image_.rows) {
            skipped_oob++;
            continue;
        }

        visible_points++;

        // Cache visible landmarks for fallback matching
        visible_lm_descs.push_back(lm->descriptor_);
        visible_lm_list.push_back(lm);
        visible_lm_pts.push_back(cv::Point3f(pos_w[0], pos_w[1], pos_w[2]));
        visible_lm_source_buckets.push_back(static_cast<LocalMapSourceBucket>(source_idx));
        
        // Search for match in current frame features with ratio test
        int best_idx = -1;
        double best_dist = 64.0;
        double second_best_dist = 256.0;
        const double search_radius_sq = 100.0 * 100.0;

        for (size_t i = 0; i < current_frame_->keypoints_.size(); ++i) {
             if (keypoint_already_matched[i]) continue;
             const auto& kp = current_frame_->keypoints_[i];
             double dist_spatial = (kp.pt.x - px[0])*(kp.pt.x - px[0]) + (kp.pt.y - px[1])*(kp.pt.y - px[1]);

             if (dist_spatial < search_radius_sq) {
                 double dist_desc = cv::norm(current_frame_->descriptors_.row(i), lm->descriptor_, cv::NORM_HAMMING);
                 if (dist_desc < best_dist) {
                     second_best_dist = best_dist;
                     best_dist = dist_desc;
                     best_idx = i;
                 } else if (dist_desc < second_best_dist) {
                     second_best_dist = dist_desc;
                 }
             }
        }

        // Ratio test: reject ambiguous matches
        if (best_idx != -1 && best_dist < 0.7 * second_best_dist) {
            object_points.push_back(cv::Point3f(pos_w[0], pos_w[1], pos_w[2]));
            image_points.push_back(current_frame_->keypoints_[best_idx].pt);
            
            matched_landmarks.push_back(lm);
            matched_kp_indices.push_back(best_idx);
            keypoint_already_matched[best_idx] = true;
            
            matches_found++;
        }
    }
    
    std::cout << "TrackLocalMap: Visible: " << visible_points
              << ", Matches: " << matches_found
              << " (skipped nonfinite=" << skipped_nonfinite
              << " behind/close=" << skipped_behind_or_close
              << " oob=" << skipped_oob
              << ")" << std::endl;
    filter_correspondences_by_pose(35.0);
    if (!object_points.empty()) {
        std::cout << "TrackLocalMap: Pose-gated matches: " << object_points.size() << std::endl;
    }

    // Fallback: if pose is too noisy, projection-based matching may find 0.
    // In that case, do global descriptor matching (landmark descriptor -> current descriptors)
    // to bootstrap PnP.
    const bool late_sparse_mono_bootstrap =
        current_frame_->depth_image_.empty() &&
        num_tracked_features_ <= 25 &&
        current_frame_->keypoints_.size() >= 700;
    if (object_points.size() < kMinBootstrapCorrespondences) {
        cv::Mat fallback_lm_descs = visible_lm_descs;
        std::vector<Landmark::Ptr> fallback_lm_list = visible_lm_list;
        std::vector<cv::Point3f> fallback_lm_pts = visible_lm_pts;
        bool fallback_from_all_landmarks = false;
        const std::size_t bootstrap_preexisting_matches = object_points.size();
        std::vector<LocalMapSourceBucket> fallback_lm_buckets;
        // Mono uses a floor of 75; RGB-D (which can rebuild correspondences from depth) uses 80.
        const bool mono_no_depth = current_frame_->depth_image_.empty();
        const std::size_t fallback_visible_pool_floor = mono_no_depth ? 75U : 80U;
        if (fallback_lm_list.size() < fallback_visible_pool_floor &&
            landmarks.size() > fallback_lm_list.size()) {
            fallback_lm_descs = cv::Mat();
            fallback_lm_list.clear();
            fallback_lm_pts.clear();
            fallback_lm_buckets.clear();
            for (std::size_t li = 0; li < landmarks.size(); ++li) {
                const auto& lm = landmarks[li];
                if (!lm || lm->descriptor_.empty()) continue;
                const Vec3 pos_w = lm->getPos();
                if (!std::isfinite(pos_w[0]) || !std::isfinite(pos_w[1]) || !std::isfinite(pos_w[2])) continue;
                fallback_lm_descs.push_back(lm->descriptor_);
                fallback_lm_list.push_back(lm);
                fallback_lm_pts.emplace_back(pos_w[0], pos_w[1], pos_w[2]);
                const std::size_t sb = li < landmark_source_buckets.size()
                    ? static_cast<std::size_t>(landmark_source_buckets[li])
                    : static_cast<std::size_t>(LocalMapSourceBucket::GlobalFallback);
                fallback_lm_buckets.push_back(static_cast<LocalMapSourceBucket>(sb));
            }
            fallback_from_all_landmarks = true;
        } else {
            fallback_lm_buckets = visible_lm_source_buckets;
        }

        if (!fallback_lm_descs.empty() && !current_frame_->descriptors_.empty()) {
            cv::BFMatcher bf(cv::NORM_HAMMING);
            std::vector<std::vector<cv::DMatch>> knn;
            bf.knnMatch(fallback_lm_descs, current_frame_->descriptors_, knn, 2);

            std::vector<bool> kp_used = keypoint_already_matched;
            std::vector<bool> lm_used(fallback_lm_list.size(), false);

            struct MatchCandidate {
                int lm_idx;
                int kp_idx;
                float dist;
                int octave;
                LocalMapSourceBucket source_bucket;
                bool coarse_ok = false;
            };
            std::vector<MatchCandidate> candidates;

            for (const auto& ms : knn) {
                if (ms.size() < 2) continue;
                const auto& m1 = ms[0];
                const auto& m2 = ms[1];

                // Distance/ratio gates: matches that pass both feed the candidate list.
                if (m1.distance > 65.0f) {
                    continue;
                }
                if (m1.distance >= 0.75f * m2.distance) {
                    continue;
                }

                if (m1.queryIdx < 0 || m1.queryIdx >= static_cast<int>(fallback_lm_list.size()) ||
                    m1.trainIdx < 0 || m1.trainIdx >= static_cast<int>(current_frame_->keypoints_.size())) {
                    continue;
                }
                if (lm_used[m1.queryIdx] || kp_used[m1.trainIdx]) {
                    continue;
                }

                const int kp_idx = m1.trainIdx;
                const int octave =
                    (kp_idx >= 0 && kp_idx < static_cast<int>(current_frame_->keypoints_.size()))
                        ? current_frame_->keypoints_[kp_idx].octave
                        : 0;
                LocalMapSourceBucket sb = LocalMapSourceBucket::GlobalFallback;
                if (m1.queryIdx >= 0 &&
                    m1.queryIdx < static_cast<int>(fallback_lm_buckets.size())) {
                    sb = fallback_lm_buckets[static_cast<std::size_t>(m1.queryIdx)];
                }
                candidates.push_back({m1.queryIdx, kp_idx, m1.distance, octave, sb, false});
            }

            const SE3 T_cw_boot = current_frame_->getPose();
            auto bootstrap_coarse_ok = [&](int lm_idx, int kp_idx) -> bool {
                if (lm_idx < 0 || lm_idx >= static_cast<int>(fallback_lm_pts.size()) || kp_idx < 0 ||
                    kp_idx >= static_cast<int>(current_frame_->keypoints_.size())) {
                    return false;
                }
                const auto& Pw = fallback_lm_pts[lm_idx];
                Vec3 p_w(Pw.x, Pw.y, Pw.z);
                Vec3 p_c = T_cw_boot * p_w;
                if (!std::isfinite(p_c.x()) || !std::isfinite(p_c.y()) || !std::isfinite(p_c.z())) {
                    return false;
                }
                if (p_c[2] <= kMinTrackedDepthMeters || p_c[2] > kMaxTrackedDepthMeters) {
                    return false;
                }
                const int oct = current_frame_->keypoints_[kp_idx].octave;
                const double gate_px =
                    55.0 * (1.0 + 0.12 * static_cast<double>(std::max(0, oct)));
                Vec2 proj = current_frame_->camera_->project(p_c);
                const auto& uv = current_frame_->keypoints_[kp_idx].pt;
                const double dx = uv.x - proj[0];
                const double dy = uv.y - proj[1];
                return (dx * dx + dy * dy) <= gate_px * gate_px;
            };

            // For late sparse mono recovery, prefer candidates that are already most
            // geometrically consistent with the current pose estimate, then fall back to
            // descriptor strength. Bucket source remains a deterministic tie-break only:
            // ordering bucket before descriptor distance regressed room_mono ATE.
            auto fallback_bucket_rank = [](LocalMapSourceBucket b) -> int {
                switch (b) {
                    case LocalMapSourceBucket::Reference:
                        return 0;
                    case LocalMapSourceBucket::PreviousReference:
                        return 1;
                    case LocalMapSourceBucket::ReferenceNeighbor:
                        return 2;
                    case LocalMapSourceBucket::PreviousReferenceNeighbor:
                        return 3;
                    case LocalMapSourceBucket::GlobalFallback:
                        return 4;
                    case LocalMapSourceBucket::Count:
                        break;
                }
                return 5;
            };
            constexpr float kDistTieEps = 1e-4f;

            if (late_sparse_mono_bootstrap) {
                for (auto& c : candidates) {
                    c.coarse_ok =
                        bootstrap_coarse_ok(c.lm_idx, c.kp_idx);
                }
                std::sort(candidates.begin(), candidates.end(),
                          [&](const MatchCandidate& a, const MatchCandidate& b) {
                              if (a.coarse_ok != b.coarse_ok) {
                                  return a.coarse_ok > b.coarse_ok;
                              }
                              if (std::abs(a.dist - b.dist) > kDistTieEps) {
                                  return a.dist < b.dist;
                              }
                              const int ra = fallback_bucket_rank(a.source_bucket);
                              const int rb = fallback_bucket_rank(b.source_bucket);
                              if (ra != rb) {
                                  return ra < rb;
                              }
                              if (a.lm_idx != b.lm_idx) {
                                  return a.lm_idx < b.lm_idx;
                              }
                              return a.kp_idx < b.kp_idx;
                          });
            } else {
                std::sort(candidates.begin(), candidates.end(), [](const MatchCandidate& a,
                                                                   const MatchCandidate& b) {
                    if (a.dist != b.dist) {
                        return a.dist < b.dist;
                    }
                    if (a.lm_idx != b.lm_idx) {
                        return a.lm_idx < b.lm_idx;
                    }
                    return a.kp_idx < b.kp_idx;
                });
            }

            for (size_t i = 0; i < candidates.size() && i < kMaxBootstrapMatches; ++i) {
                const auto& c = candidates[i];
                if (lm_used[c.lm_idx] || kp_used[c.kp_idx]) continue;
                object_points.push_back(fallback_lm_pts[c.lm_idx]);
                image_points.push_back(current_frame_->keypoints_[c.kp_idx].pt);
                matched_landmarks.push_back(fallback_lm_list[c.lm_idx]);
                matched_kp_indices.push_back(c.kp_idx);
                lm_used[c.lm_idx] = true;
                kp_used[c.kp_idx] = true;
            }
            if (fallback_from_all_landmarks) {
                std::cout << "TrackLocalMap: Descriptor bootstrap using all "
                          << fallback_lm_list.size() << " landmarks" << std::endl;
            }

            const double fallback_gate_px = fallback_from_all_landmarks ? 180.0 : 55.0;
            const double relaxed_fallback_gate_px = fallback_from_all_landmarks ? 260.0 : 85.0;
            const auto object_points_before_pose_filter = object_points;
            const auto image_points_before_pose_filter = image_points;
            const auto matched_landmarks_before_pose_filter = matched_landmarks;
            const auto matched_kp_indices_before_pose_filter = matched_kp_indices;

            PoseFilterStats pose_filter_stats =
                filter_correspondences_by_pose(fallback_gate_px, bootstrap_preexisting_matches);
            if (late_sparse_mono_bootstrap &&
                object_points.size() < kMinBootstrapCorrespondences &&
                pose_filter_stats.focus_reject_reprojection > 0) {
                object_points = object_points_before_pose_filter;
                image_points = image_points_before_pose_filter;
                matched_landmarks = matched_landmarks_before_pose_filter;
                matched_kp_indices = matched_kp_indices_before_pose_filter;
                pose_filter_stats =
                    filter_correspondences_by_pose(relaxed_fallback_gate_px, bootstrap_preexisting_matches);
            }

            used_global_fallback = true;
            std::cout << "TrackLocalMap: Fallback global matches: " << object_points.size() << std::endl;
        }
    }
    
    if (object_points.size() < 10) return false;

    // Diagnostics: check correspondence sanity before PnP.
    // This helps identify issues like broken 3D points, bad scale, or inconsistent 3D-2D pairs.
    {
        size_t n = object_points.size();
        size_t nan_or_inf = 0;
        size_t non_positive_depth = 0;

        double z_min = std::numeric_limits<double>::infinity();
        double z_max = -std::numeric_limits<double>::infinity();
        double z_sum = 0.0;
        size_t z_cnt = 0;

        double err_min = std::numeric_limits<double>::infinity();
        double err_max = -std::numeric_limits<double>::infinity();
        double err_sum = 0.0;
        size_t err_cnt = 0;

        const SE3 T_cw_est = current_frame_->getPose();

        for (size_t i = 0; i < n; ++i) {
            const auto& Pw = object_points[i];
            if (!std::isfinite(Pw.x) || !std::isfinite(Pw.y) || !std::isfinite(Pw.z)) {
                nan_or_inf++;
                continue;
            }

            Vec3 p_w(Pw.x, Pw.y, Pw.z);
            Vec3 p_c = T_cw_est * p_w;

            const double z = p_c[2];
            if (!std::isfinite(z)) {
                nan_or_inf++;
                continue;
            }
            if (z <= 0.0) {
                non_positive_depth++;
                continue;
            }

            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
            z_sum += z;
            z_cnt++;

            Vec2 proj = current_frame_->camera_->project(p_c);
            const auto& uv = image_points[i];
            const double dx = uv.x - proj[0];
            const double dy = uv.y - proj[1];
            const double e = std::sqrt(dx * dx + dy * dy);
            if (std::isfinite(e)) {
                err_min = std::min(err_min, e);
                err_max = std::max(err_max, e);
                err_sum += e;
                err_cnt++;
            }
        }

        const double z_mean = (z_cnt > 0) ? (z_sum / static_cast<double>(z_cnt)) : std::numeric_limits<double>::quiet_NaN();
        const double err_mean = (err_cnt > 0) ? (err_sum / static_cast<double>(err_cnt)) : std::numeric_limits<double>::quiet_NaN();

        std::cout << "TrackLocalMap: CorrStats n=" << n
                  << " nan_inf=" << nan_or_inf
                  << " nonpos_z=" << non_positive_depth
                  << " z[min/mean/max]=" << z_min << "/" << z_mean << "/" << z_max
                  << " reproj_err_px[min/mean/max]=" << err_min << "/" << err_mean << "/" << err_max
                  << " used_global_fallback=" << (used_global_fallback ? 1 : 0)
                  << std::endl;
    }
    
    // 3. Optimize Pose (solvePnPRansac)
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    
    // Initial guess from motion model
    Eigen::Vector3d t = current_frame_->getPose().translation();
    Eigen::Matrix3d R = current_frame_->getPose().rotationMatrix();
    cv::Mat R_cv, t_cv;
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    cv::eigen2cv(t, tvec);
    
    // If correspondences come from global descriptor matching, do not trust the motion-model pose as an initial guess.
    const bool use_extrinsic_guess = !used_global_fallback;
    const int pnp_ransac_iterations = 150;
    const double pnp_ransac_reproj_px = 10.0;
    const double pnp_refine_gate_px = 8.0;

    enum class PnpMethod { EPNP, P3P, ITERATIVE };
    auto try_pnp = [&](PnpMethod method) -> bool {
        int flag = cv::SOLVEPNP_EPNP;
        const char* name = "EPNP";
        if (method == PnpMethod::P3P) {
            flag = cv::SOLVEPNP_P3P;
            name = "P3P";
        } else if (method == PnpMethod::ITERATIVE) {
            flag = cv::SOLVEPNP_ITERATIVE;
            name = "ITERATIVE";
        }

        std::vector<int> tmp_inliers;
        cv::Mat rvec_tmp = rvec.clone();
        cv::Mat tvec_tmp = tvec.clone();

        bool ok = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                     rvec_tmp, tvec_tmp, use_extrinsic_guess,
                                     pnp_ransac_iterations, pnp_ransac_reproj_px, 0.995, tmp_inliers, flag);
        if (ok) {
            rvec = rvec_tmp;
            tvec = tvec_tmp;
            inliers.swap(tmp_inliers);
            std::cout << "TrackLocalMap: PnP method=" << name << " inliers=" << inliers.size() << std::endl;
            return true;
        }
        return false;
    };

    bool success = try_pnp(PnpMethod::EPNP) || try_pnp(PnpMethod::P3P) || try_pnp(PnpMethod::ITERATIVE);
    if (success) {
        success = refinePnPInliers(
            current_frame_,
            object_points,
            image_points,
            inliers,
            [&](const int corr_idx) { return matched_kp_indices[corr_idx]; },
            pnp_refine_gate_px,
            rvec,
            tvec,
            inliers);
    }

    if (success && inliers.size() >= kMinTrackLocalMapInliers) {
        std::cout << "TrackLocalMap: PnP Success, inliers: " << inliers.size() << std::endl;
        const SE3 new_pose = poseFromOpenCvPose(rvec, tvec);

        if (!isCameraPositionWithinBounds(new_pose)) {
            const Vec3 cam_pos = new_pose.inverse().translation();
            std::cout << "TrackLocalMap: REJECTED - Absolute position out of bounds: "
                      << cam_pos.transpose() << std::endl;
            return false;
        }

        if (last_frame_) {
            const PoseChange pose_change =
                computePoseChange(new_pose, last_frame_->getPose());
            const double max_trans_change = 0.5;
            const double max_rot_change = 0.6;

            std::cout << "TrackLocalMap: Pose change - trans=" << pose_change.translation
                      << " rot=" << pose_change.rotation << " rad" << std::endl;

            if (!shouldAcceptLocalMapPoseUpdate(inliers.size(),
                                                prior_support,
                                                used_global_fallback,
                                                pose_change.translation,
                                                pose_change.rotation,
                                                recovery_state_.stabilization_frames_remaining)) {
                num_tracked_features_ = static_cast<int>(prior_support);
                std::cout << "TrackLocalMap: REJECTED - Recovery stabilization kept prior pose"
                          << " support=" << inliers.size()
                          << " prior_support=" << prior_support
                          << " trans=" << pose_change.translation
                          << " rot=" << pose_change.rotation
                          << " fallback=" << (used_global_fallback ? 1 : 0)
                          << " window=" << recovery_state_.stabilization_frames_remaining
                          << std::endl;
                return true;
            }

            if (pose_change.translation > max_trans_change ||
                pose_change.rotation > max_rot_change) {
                std::cout << "TrackLocalMap: REJECTED - Pose change too large! "
                          << "trans=" << pose_change.translation
                          << " (max=" << max_trans_change << ") "
                          << "rot=" << pose_change.rotation
                          << " (max=" << max_rot_change << ")" << std::endl;
                return false;
            }
        }

        current_frame_->setPose(new_pose);
        assignFrameLandmarksFromInliers(current_frame_, matched_kp_indices,
                                        matched_landmarks, inliers);
        num_tracked_features_ = static_cast<int>(inliers.size());

        return true;
    }

    if (success && inliers.size() < kMinTrackLocalMapInliers) {
        std::cout << "TrackLocalMap: PnP rejected - insufficient inliers: " << inliers.size()
                  << " (min=" << kMinTrackLocalMapInliers << ")" << std::endl;
    } else {
        std::cout << "TrackLocalMap: PnP failed. correspondences=" << object_points.size()
                  << " used_global_fallback=" << (used_global_fallback ? 1 : 0) << std::endl;
    }

    return false;
}

bool Tracking::trackReferenceKeyframe() {
    if (!last_frame_) return false;

    // Compute matches between current and last frame
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(current_frame_->descriptors_, last_frame_->descriptors_, knn, 2);

    struct MatchCandidate {
        int query_idx;
        int train_idx;
        float dist;
    };
    std::vector<MatchCandidate> candidates;
    candidates.reserve(knn.size());

    for (const auto& ms : knn) {
        if (ms.size() < 2) continue;
        const auto& m1 = ms[0];
        const auto& m2 = ms[1];
        // Favor cleaner frame-to-frame propagation for monocular tracking.
        if (m1.distance > 65.0f) continue;
        if (m1.distance >= 0.75f * m2.distance) continue;
        candidates.push_back({m1.queryIdx, m1.trainIdx, m1.distance});
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        if (a.query_idx != b.query_idx) return a.query_idx < b.query_idx;
        return a.train_idx < b.train_idx;
    });

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(candidates.size());
    std::vector<bool> used_query(current_frame_->keypoints_.size(), false);
    std::vector<bool> used_train(last_frame_->keypoints_.size(), false);
    for (const auto& c : candidates) {
        if (c.query_idx < 0 || c.query_idx >= static_cast<int>(current_frame_->keypoints_.size())) continue;
        if (c.train_idx < 0 || c.train_idx >= static_cast<int>(last_frame_->keypoints_.size())) continue;
        if (used_query[c.query_idx] || used_train[c.train_idx]) continue;
        used_query[c.query_idx] = true;
        used_train[c.train_idx] = true;
        good_matches.push_back(cv::DMatch(c.query_idx, c.train_idx, c.dist));
    }

    std::cout << "Matches with last frame: " << good_matches.size() << std::endl;

    if (good_matches.size() < 8) {
        return false;
    }
    
    // Propagate landmark associations from last frame via feature matches.
    // This is critical for bootstrapping 3D-2D PnP in subsequent frames.
    // Hold current_frame_->mutex_ across assign + inner writes: onBACompleted
    // may snapshot landmarks_ concurrently. last_frame_ is only read and
    // only the tracking thread writes it, so reads below are safe.
    int propagated = 0;

    // Optimization: Pose from 3D-2D
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<int> current_kp_indices;
    std::vector<Landmark::Ptr> propagated_landmarks;

    {
        std::lock_guard<std::mutex> lock(current_frame_->mutex_);
        current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);

        for (const auto& m : good_matches) {
            // Query is current, Train is last
            int idx_last = m.trainIdx;
            int idx_curr = m.queryIdx;

            if (idx_last >= 0 && idx_last < static_cast<int>(last_frame_->landmarks_.size()) &&
                last_frame_->landmarks_[idx_last]) {
                // Found a map point
                Vec3 pos = last_frame_->landmarks_[idx_last]->getPos();
                if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) continue;

                Vec3 p_c = current_frame_->getPose() * pos;
                if (!std::isfinite(p_c.x()) || !std::isfinite(p_c.y()) || !std::isfinite(p_c.z())) continue;
                if (p_c[2] <= kMinTrackedDepthMeters || p_c[2] > kMaxTrackedDepthMeters) continue;

                const int octave = current_frame_->keypoints_[idx_curr].octave;
                const double gate_px = 48.0 * (1.0 + 0.12 * static_cast<double>(std::max(0, octave)));
                Vec2 proj = current_frame_->camera_->project(p_c);
                const auto& uv = current_frame_->keypoints_[idx_curr].pt;
                const double dx = uv.x - proj[0];
                const double dy = uv.y - proj[1];
                if ((dx * dx + dy * dy) > gate_px * gate_px) continue;

                object_points.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
                image_points.push_back(uv);

                current_frame_->landmarks_[idx_curr] = last_frame_->landmarks_[idx_last];
                current_kp_indices.push_back(idx_curr);
                propagated_landmarks.push_back(last_frame_->landmarks_[idx_last]);
                propagated++;
            }
        }
    }

    if (propagated > 0) {
        std::cout << "TrackReferenceKeyframe: Propagated landmarks: " << propagated << std::endl;
    }

    std::cout << "TrackReferenceKeyframe: 3D-2D correspondences: " << object_points.size() << std::endl;
    
    if (object_points.size() >= 10) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        
        // Initial guess
        Eigen::Vector3d t = current_frame_->getPose().translation();
        Eigen::Matrix3d R = current_frame_->getPose().rotationMatrix();
        cv::Mat R_cv, t_cv;
        cv::eigen2cv(R, R_cv);
        cv::Rodrigues(R_cv, rvec);
        cv::eigen2cv(t, tvec);

        bool success = cv::solvePnPRansac(object_points, image_points, current_frame_->camera_->K(), cv::Mat(),
                                          rvec, tvec, true, 250, 8.0, 0.995, inliers, cv::SOLVEPNP_EPNP);

        if (success) {
            success = refinePnPInliers(
                current_frame_,
                object_points,
                image_points,
                inliers,
                [&](const int corr_idx) { return current_kp_indices[corr_idx]; },
                6.0,
                rvec,
                tvec,
                inliers);
        }

        if (success && inliers.size() >= kMinTrackReferenceInliers) {
             std::cout << "TrackReferenceKeyframe: PnP Success, inliers: "
                       << inliers.size() << std::endl;
             const SE3 new_pose = poseFromOpenCvPose(rvec, tvec);

             if (!isCameraPositionWithinBounds(new_pose)) {
                 const Vec3 cam_pos = new_pose.inverse().translation();
                 std::cout << "TrackReferenceKeyframe: REJECTED - Absolute position out of bounds: "
                           << cam_pos.transpose() << std::endl;
                 return false;
             }

             bool accept_pose = true;
             if (last_frame_) {
                 const PoseChange pose_change =
                     computePoseChange(new_pose, last_frame_->getPose());
                 const double max_trans_change = 0.35;
                 const double max_rot_change = 0.45;

                 std::cout << "TrackReferenceKeyframe: Pose change - trans="
                           << pose_change.translation
                           << " rot=" << pose_change.rotation << " rad" << std::endl;

                 accept_pose = pose_change.translation <= max_trans_change &&
                               pose_change.rotation <= max_rot_change;
                 if (!accept_pose) {
                     std::cout << "TrackReferenceKeyframe: REJECTED - Pose change too large!"
                               << std::endl;
                 }
             }

             if (accept_pose) {
                 assignFrameLandmarksFromInliers(current_frame_, current_kp_indices,
                                                 propagated_landmarks, inliers);
                 current_frame_->setPose(new_pose);
                 return true;
             }
        }
    }

    // If PnP fails (e.g. no 3D points in last frame yet), we rely on motion model.
    // But since we are here, we probably have some tracking.

    return true;
}

bool Tracking::shouldAcceptRecomputedPose(double err_before, double err_after) {
    if (!std::isfinite(err_before) || !std::isfinite(err_after)) {
        return false;
    }
    return err_after < err_before;
}

bool Tracking::shouldAcceptLocalMapPoseUpdate(std::size_t support,
                                              std::size_t prior_support,
                                              bool used_global_fallback,
                                              double trans_change,
                                              double rot_change,
                                              int stabilization_frames_remaining) {
    if (stabilization_frames_remaining <= 0) {
        return true;
    }
    if (!std::isfinite(trans_change) || !std::isfinite(rot_change)) {
        return false;
    }

    const bool thin_support = support < min_stable_support_ || used_global_fallback;
    const bool support_regressed =
        prior_support > 0 && (support * 4 < prior_support * 3);
    const double max_recovery_trans_change =
        (thin_support || support_regressed) ? recovery_max_change_strict_ : recovery_max_change_relaxed_;
    const double max_recovery_rot_change =
        (thin_support || support_regressed) ? recovery_max_change_strict_ : recovery_max_change_relaxed_;

    return trans_change <= max_recovery_trans_change &&
           rot_change <= max_recovery_rot_change;
}

bool Tracking::shouldConsiderRelocalizationCandidate(double distance_to_anchor,
                                                     bool is_reference_candidate,
                                                     bool pending_loop_correction,
                                                     int stabilization_frames_remaining) {
    if (is_reference_candidate) {
        return true;
    }
    if (!pending_loop_correction && stabilization_frames_remaining <= 0) {
        return true;
    }
    if (!std::isfinite(distance_to_anchor)) {
        return false;
    }

    const double max_distance = pending_loop_correction
        ? loop_relocalization_radius_m_
        : recovery_relocalization_radius_m_;
    return distance_to_anchor <= max_distance;
}

void Tracking::onBACompleted() {
    std::lock_guard<std::mutex> lock(pose_mutex_);

    if (!current_frame_) {
        return;
    }

    // Skip if current frame is a keyframe (its pose was optimized in BA)
    // Check by comparing with reference keyframe
    if (reference_keyframe_ && current_frame_->id_ == reference_keyframe_->id_) {
        return;
    }

    std::cout << "Tracking: BA completed, recomputing current frame pose..." << std::endl;
    recomputeCurrentPose();
}

std::size_t Tracking::countValidFrameLandmarks(const Frame::Ptr& frame) {
    if (!frame) {
        return 0;
    }

    // Snapshot under frame->mutex_ — this path is reached from onBACompleted
    // on the LocalMapping thread while the tracking thread may concurrently
    // write frame->landmarks_[i].
    const auto snapshot = frame->snapshotLandmarks();
    std::size_t correspondences = 0;
    for (const auto& lm : snapshot) {
        if (!lm || lm->isBad()) continue;
        const Vec3 pos = lm->getPos();
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
            continue;
        }
        ++correspondences;
    }
    return correspondences;
}

bool Tracking::applyPendingLoopCorrection(const char* phase) {
    std::lock_guard<std::mutex> lock(pose_mutex_);

    if (!loop_correction_state_.pending) {
        return false;
    }
    if (map_ && map_->loop_correcting_.load()) {
        return false;
    }
    if (!current_frame_ || !current_frame_->camera_) {
        return false;
    }

    const std::size_t correspondences = countValidFrameLandmarks(current_frame_);

    const auto expire_pending_loop_correction = [&](const char* message) {
        loop_correction_state_.pending = false;
        loop_correction_state_.pending_deferrals = 0;
        velocity_ = SE3();
        loop_correction_state_.force_keyframe_insertion_once = true;
        loop_correction_state_.force_reference_refresh_once = true;
        recovery_state_.stabilization_frames_remaining =
            recovery_stabilization_window_frames_;
        std::cout << message << std::endl;
    };

    if (correspondences < min_loop_correction_correspondences_) {
        ++loop_correction_state_.pending_deferrals;
        std::cout << "Tracking: Pending loop correction deferred at " << phase
                  << " (pairs=" << correspondences
                  << ", min_pairs=" << min_loop_correction_correspondences_
                  << ", deferral=" << loop_correction_state_.pending_deferrals
                  << "/" << max_loop_correction_deferrals_ << ")" << std::endl;
        if (loop_correction_state_.pending_deferrals >= max_loop_correction_deferrals_) {
            expire_pending_loop_correction(
                "Tracking: Pending loop correction expired, reset velocity only");
        }
        return false;
    }

    std::cout << "Tracking: Applying pending loop correction at " << phase
              << " with pairs=" << correspondences << std::endl;
    if (recomputeCurrentPose()) {
        loop_correction_state_.pending = false;
        loop_correction_state_.pending_deferrals = 0;
        velocity_ = SE3();
        loop_correction_state_.skip_velocity_update_once = true;
        recovery_state_.stabilization_frames_remaining =
            recovery_stabilization_window_frames_;
        return true;
    }

    ++loop_correction_state_.pending_deferrals;
    std::cout << "Tracking: Pending loop correction retained after failed recompute at " << phase
              << " (deferral=" << loop_correction_state_.pending_deferrals
              << "/" << max_loop_correction_deferrals_ << ")" << std::endl;
    if (loop_correction_state_.pending_deferrals >= max_loop_correction_deferrals_) {
        expire_pending_loop_correction(
            "Tracking: Pending loop correction expired after failed recomputes; forcing reference refresh");
    }
    return false;
}

bool Tracking::recomputeCurrentPose() {
    // Recompute current frame pose using updated landmark positions from BA
    if (!current_frame_ || !current_frame_->camera_) {
        return false;
    }

    // Collect 3D-2D correspondences from current frame's landmark associations.
    // Snapshot landmarks_ under current_frame_->mutex_ — this runs on the
    // LocalMapping thread via onBACompleted while tracking may be writing.
    const auto landmarks_snapshot = current_frame_->snapshotLandmarks();
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> indices;

    for (size_t i = 0; i < landmarks_snapshot.size(); i++) {
        auto lm = landmarks_snapshot[i];
        if (!lm || lm->isBad()) continue;

        Vec3 pos = lm->getPos();  // BA-updated position
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
            continue;
        }

        pts3d.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
        pts2d.push_back(current_frame_->keypoints_[i].pt);
        indices.push_back(static_cast<int>(i));
    }

    std::cout << "Tracking::recomputeCurrentPose: 3D-2D pairs = " << pts3d.size() << std::endl;

    if (pts3d.size() < kMinPoseRecomputeCorrespondences) {
        std::cout << "Tracking::recomputeCurrentPose: Not enough correspondences" << std::endl;
        return false;
    }

    // PnP to recompute pose
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(pts3d, pts2d, current_frame_->camera_->K(), cv::Mat(),
                                  rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

    if (ok && inliers.size() >= kMinPoseRecomputeInliers) {
        const SE3 new_pose = poseFromOpenCvPose(rvec, tvec);
        const SE3 old_pose = current_frame_->getPose();
        int valid_count = 0;
        const double err_before = computeAverageReprojectionError(
            current_frame_, old_pose, pts3d, pts2d, inliers, &valid_count);
        const double err_after = computeAverageReprojectionError(
            current_frame_, new_pose, pts3d, pts2d, inliers);

        std::cout << "Tracking::recomputeCurrentPose: Reprojection error before=" << err_before
                  << " after=" << err_after
                  << " inliers=" << inliers.size()
                  << " valid=" << valid_count << std::endl;

        if (shouldAcceptRecomputedPose(err_before, err_after)) {
            current_frame_->setPose(new_pose);

            // Also update velocity model
            if (last_frame_) {
                velocity_ = current_frame_->getPose() * last_frame_->getPose().inverse();
            }

            std::cout << "Tracking::recomputeCurrentPose: Pose updated successfully" << std::endl;
            return true;
        }
        std::cout << "Tracking::recomputeCurrentPose: New pose rejected (no reprojection improvement)"
                  << std::endl;
        return false;
    }

    std::cout << "Tracking::recomputeCurrentPose: PnP failed or insufficient inliers ("
              << inliers.size() << ")" << std::endl;
    return false;
}

void Tracking::onLoopCorrected() {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    loop_correction_state_.pending = true;
    loop_correction_state_.pending_deferrals = 0;
    std::cout << "Tracking: Loop correction completed, pose recompute queued for tracking thread" << std::endl;
}

bool Tracking::relocalize() {
    if (!map_) return false;

    // Get all keyframes from map
    auto keyframes = map_->getAllKeyframes();
    if (keyframes.empty()) {
        std::cout << "Relocalize: No keyframes available" << std::endl;
        return false;
    }

    const cv::Mat& curr_desc = current_frame_->descriptors_;
    if (curr_desc.empty()) {
        std::cout << "Relocalize: Current frame has no descriptors" << std::endl;
        return false;
    }

    // Structure to hold candidate keyframes with their match scores
    struct Candidate {
        Keyframe::Ptr kf;
        std::vector<cv::DMatch> matches;
        int score;
        int valid_3d_matches;
        double distance_to_anchor = std::numeric_limits<double>::infinity();
        bool local_to_anchor = false;
    };
    std::vector<Candidate> candidates;

    const bool prefer_local_candidates =
        loop_correction_state_.pending || recovery_state_.stabilization_frames_remaining > 0;
    const Vec3 anchor_position = recovery_state_.last_good_pose.inverse().translation();
    const bool anchor_valid = anchor_position.allFinite();
    bool have_local_candidate = false;

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Match current frame with each keyframe
    for (auto& kf_pair : keyframes) {
        auto& kf = kf_pair.second;
        if (!kf || kf->descriptors_.empty()) continue;

        const bool is_reference_candidate =
            kf == reference_keyframe_ || kf == previous_reference_keyframe_;
        double distance_to_anchor = std::numeric_limits<double>::infinity();
        if (prefer_local_candidates && anchor_valid) {
            const Vec3 candidate_position = kf->T_cw_.inverse().translation();
            if (candidate_position.allFinite()) {
                distance_to_anchor = (candidate_position - anchor_position).norm();
            }
        }
        const bool local_to_anchor = shouldConsiderRelocalizationCandidate(
            distance_to_anchor, is_reference_candidate, loop_correction_state_.pending,
            recovery_state_.stabilization_frames_remaining);

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(curr_desc, kf->descriptors_, knn, 2);

        std::vector<cv::DMatch> good;
        for (auto& m : knn) {
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance && m[0].distance < 65.0f) {
                good.push_back(m[0]);
            }
        }

        // Snapshot kf->landmarks_ under kf->mutex_ to avoid racing with
        // LocalMapping::createNewMapPoints writes on the same container.
        std::vector<Landmark::Ptr> kf_landmarks_snapshot;
        {
            std::lock_guard<std::mutex> lock(kf->mutex_);
            kf_landmarks_snapshot = kf->landmarks_;
        }

        int valid_3d_matches = 0;
        for (const auto& m : good) {
            const int kf_idx = m.trainIdx;
            if (kf_idx < 0 || kf_idx >= static_cast<int>(kf_landmarks_snapshot.size())) continue;
            const auto& lm = kf_landmarks_snapshot[kf_idx];
            if (!lm || lm->isBad()) continue;

            const Vec3 pos = lm->getPos();
            if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) continue;
            ++valid_3d_matches;
        }

        if (valid_3d_matches >= 8) {
            candidates.push_back(
                {kf, good, valid_3d_matches, valid_3d_matches, distance_to_anchor, local_to_anchor});
            have_local_candidate = have_local_candidate || local_to_anchor;
        }
    }

    if (candidates.empty()) {
        std::cout << "Relocalize: No candidate keyframes found" << std::endl;
        return false;
    }

    if (prefer_local_candidates && have_local_candidate) {
        std::cout << "Relocalize: Prioritizing local candidates during recovery"
                  << " (total_candidates=" << candidates.size() << ")" << std::endl;
    }

    // Sort by score (number of matches). On ties, prefer keyframes closer in time to the live
    // frame (smaller |id - current|) before stable id ordering — helps room revisits vs arbitrary id.
    const long current_frame_id = static_cast<long>(current_frame_->id_);
    auto temporal_id_gap = [current_frame_id](const Candidate& c) -> long {
        if (!c.kf) {
            return std::numeric_limits<long>::max();
        }
        return std::llabs(current_frame_id - static_cast<long>(c.kf->id_));
    };
    std::sort(candidates.begin(), candidates.end(),
              [&](const Candidate& a, const Candidate& b) {
                  if (a.local_to_anchor != b.local_to_anchor) {
                      return a.local_to_anchor > b.local_to_anchor;
                  }
                  if (a.score == b.score) {
                      if (a.matches.size() == b.matches.size()) {
                          if (a.distance_to_anchor != b.distance_to_anchor) {
                              return a.distance_to_anchor < b.distance_to_anchor;
                          }
                          const long gap_a = temporal_id_gap(a);
                          const long gap_b = temporal_id_gap(b);
                          if (gap_a != gap_b) {
                              return gap_a < gap_b;
                          }
                          const long a_id = a.kf ? static_cast<long>(a.kf->id_) : -1L;
                          const long b_id = b.kf ? static_cast<long>(b.kf->id_) : -1L;
                          return a_id < b_id;
                      }
                      return a.matches.size() > b.matches.size();
                  }
                  return a.score > b.score;
              });

    std::cout << "Relocalize: Found " << candidates.size() << " candidate keyframes" << std::endl;

    // Try PnP with top N candidates
    const int max_candidates = 20;
    struct SuccessfulRelocalization {
        bool found = false;
        int candidate_index = -1;
        std::size_t inlier_count = 0;
        SE3 pose = SE3();
        std::vector<int> inlier_indices;
        std::vector<int> match_indices;
        double avg_reprojection_error_px = std::numeric_limits<double>::infinity();
        int valid_reprojection_count = 0;
        double pose_change_translation = std::numeric_limits<double>::infinity();
        double pose_change_rotation = std::numeric_limits<double>::infinity();
    };
    SuccessfulRelocalization best_success;
    const bool quality_first_recovery = prefer_local_candidates && have_local_candidate;

    auto should_replace_best = [&](const Candidate& cand,
                                   std::size_t inlier_count,
                                   double avg_reprojection_error_px,
                                   double pose_change_translation,
                                   double pose_change_rotation) {
        if (!best_success.found) {
            return true;
        }

        const auto& best_candidate = candidates[best_success.candidate_index];
        if (cand.local_to_anchor != best_candidate.local_to_anchor) {
            return cand.local_to_anchor > best_candidate.local_to_anchor;
        }

        auto compare_smaller_with_margin = [](double lhs, double rhs, double margin) {
            const bool lhs_finite = std::isfinite(lhs);
            const bool rhs_finite = std::isfinite(rhs);
            if (lhs_finite != rhs_finite) {
                return lhs_finite ? -1 : 1;
            }
            if (!lhs_finite) {
                return 0;
            }
            if (lhs + margin < rhs) {
                return -1;
            }
            if (rhs + margin < lhs) {
                return 1;
            }
            return 0;
        };

        if (quality_first_recovery) {
            const std::size_t inlier_gap =
                inlier_count > best_success.inlier_count
                    ? inlier_count - best_success.inlier_count
                    : best_success.inlier_count - inlier_count;
            if (inlier_gap <= 1) {
                const int trans_cmp = compare_smaller_with_margin(
                    pose_change_translation, best_success.pose_change_translation, 0.03);
                if (trans_cmp != 0) {
                    return trans_cmp < 0;
                }

                const int rot_cmp = compare_smaller_with_margin(
                    pose_change_rotation, best_success.pose_change_rotation, 0.01);
                if (rot_cmp != 0) {
                    return rot_cmp < 0;
                }

                const int reproj_cmp = compare_smaller_with_margin(
                    avg_reprojection_error_px, best_success.avg_reprojection_error_px, 1.0);
                if (reproj_cmp != 0) {
                    return reproj_cmp < 0;
                }
            }
        }

        if (inlier_count != best_success.inlier_count) {
            return inlier_count > best_success.inlier_count;
        }
        if (cand.valid_3d_matches != best_candidate.valid_3d_matches) {
            return cand.valid_3d_matches > best_candidate.valid_3d_matches;
        }
        if (avg_reprojection_error_px != best_success.avg_reprojection_error_px) {
            return avg_reprojection_error_px < best_success.avg_reprojection_error_px;
        }
        if (pose_change_translation != best_success.pose_change_translation) {
            return pose_change_translation < best_success.pose_change_translation;
        }
        if (pose_change_rotation != best_success.pose_change_rotation) {
            return pose_change_rotation < best_success.pose_change_rotation;
        }
        if (cand.score != best_candidate.score) {
            return cand.score > best_candidate.score;
        }
        if (cand.distance_to_anchor != best_candidate.distance_to_anchor) {
            return cand.distance_to_anchor < best_candidate.distance_to_anchor;
        }

        const long cand_id = cand.kf ? static_cast<long>(cand.kf->id_) : -1L;
        const long best_id = best_candidate.kf ? static_cast<long>(best_candidate.kf->id_) : -1L;
        return cand_id < best_id;
    };

    for (int i = 0; i < std::min(static_cast<int>(candidates.size()), max_candidates); i++) {
        auto& cand = candidates[i];

        // Build 3D-2D correspondences
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        std::vector<int> match_indices;

        // Snapshot cand.kf->landmarks_ under kf->mutex_ — LocalMapping's
        // createNewMapPoints writes kf->landmarks_[idx] concurrently and TSan
        // flagged the unprotected read here.
        std::vector<Landmark::Ptr> kf_landmarks;
        {
            std::lock_guard<std::mutex> lock(cand.kf->mutex_);
            kf_landmarks = cand.kf->landmarks_;
        }

        for (size_t m_idx = 0; m_idx < cand.matches.size(); m_idx++) {
            auto& m = cand.matches[m_idx];
            int kf_idx = m.trainIdx;
            int curr_idx = m.queryIdx;

            if (kf_idx >= 0 && kf_idx < static_cast<int>(kf_landmarks.size()) &&
                kf_landmarks[kf_idx]) {
                auto lm = kf_landmarks[kf_idx];
                if (lm->isBad()) continue;

                Vec3 pos = lm->getPos();
                if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
                    continue;
                }

                pts3d.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
                pts2d.push_back(current_frame_->keypoints_[curr_idx].pt);
                match_indices.push_back(static_cast<int>(m_idx));
            }
        }

        if (pts3d.size() < 10) {
            continue;
        }

        // PnP with RANSAC
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(pts3d, pts2d, current_frame_->camera_->K(), cv::Mat(),
                                      rvec, tvec, false, 500, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

        if (ok && inliers.size() >= 10) {
            std::vector<cv::Point3f> refine_pts3d;
            std::vector<cv::Point2f> refine_pts2d;
            refine_pts3d.reserve(inliers.size());
            refine_pts2d.reserve(inliers.size());
            for (int idx : inliers) {
                if (idx < 0 || idx >= static_cast<int>(pts3d.size())) continue;
                refine_pts3d.push_back(pts3d[idx]);
                refine_pts2d.push_back(pts2d[idx]);
            }

            if (refine_pts3d.size() >= 6) {
                cv::Mat rvec_refined = rvec.clone();
                cv::Mat tvec_refined = tvec.clone();
                bool refined = cv::solvePnP(refine_pts3d, refine_pts2d, current_frame_->camera_->K(),
                                            cv::Mat(), rvec_refined, tvec_refined, true, cv::SOLVEPNP_ITERATIVE);
                if (refined) {
                    rvec = rvec_refined;
                    tvec = tvec_refined;
                }
            }

            // Success! Validate pose before accepting
            const SE3 candidate_pose = poseFromOpenCvPose(rvec, tvec);
            if (!candidate_pose.translation().allFinite() ||
                !candidate_pose.unit_quaternion().coeffs().allFinite()) {
                std::cout << "Relocalize: Rejected KF " << cand.kf->id_
                          << " - non-finite pose estimate" << std::endl;
                continue;
            }

            const Vec3 cam_pos = candidate_pose.inverse().translation();
            if (!cam_pos.allFinite() || cam_pos.norm() > 100.0) {
                std::cout << "Relocalize: Rejected KF " << cand.kf->id_
                          << " - implausible camera position norm=" << cam_pos.norm()
                          << std::endl;
                continue;
            }
            // Sanity check: camera position should be near the candidate KF
            SE3 T_wc_cand = cand.kf->T_cw_.inverse();
            SE3 T_wc_new = candidate_pose.inverse();
            double dist_to_kf = (T_wc_new.translation() - T_wc_cand.translation()).norm();
            if (dist_to_kf > 5.0) {
                std::cout << "Relocalize: Rejected KF " << cand.kf->id_
                          << " - pose too far from KF (" << dist_to_kf << "m)" << std::endl;
                continue;
            }

            int valid_reprojection_count = 0;
            const double avg_reprojection_error_px = computeAverageReprojectionError(
                current_frame_, candidate_pose, pts3d, pts2d, inliers, &valid_reprojection_count);
            const PoseChange pose_change =
                computePoseChange(candidate_pose, recovery_state_.last_good_pose);

            std::cout << "Relocalize: Candidate KF " << cand.kf->id_
                      << " inliers=" << inliers.size()
                      << " valid_3d=" << cand.valid_3d_matches
                      << " reproj_px=" << avg_reprojection_error_px
                      << " reproj_valid=" << valid_reprojection_count
                      << " pose_trans=" << pose_change.translation
                      << " pose_rot=" << pose_change.rotation
                      << " dist_to_anchor=" << cand.distance_to_anchor
                      << " local=" << (cand.local_to_anchor ? 1 : 0)
                      << std::endl;

            if (should_replace_best(cand,
                                    inliers.size(),
                                    avg_reprojection_error_px,
                                    pose_change.translation,
                                    pose_change.rotation)) {
                best_success.found = true;
                best_success.candidate_index = i;
                best_success.inlier_count = inliers.size();
                best_success.pose = candidate_pose;
                best_success.inlier_indices = inliers;
                best_success.match_indices = match_indices;
                best_success.avg_reprojection_error_px = avg_reprojection_error_px;
                best_success.valid_reprojection_count = valid_reprojection_count;
                best_success.pose_change_translation = pose_change.translation;
                best_success.pose_change_rotation = pose_change.rotation;
            }
        }
    }

    if (best_success.found) {
        const auto& best_candidate = candidates[best_success.candidate_index];
        current_frame_->setPose(best_success.pose);

        // Snapshot best_candidate.kf->landmarks_ first (separate lock) so we
        // don't hold two container mutexes at once — LocalMapping only takes
        // one KF mutex at a time, and the existing race-fix convention avoids
        // lock-order inversion.
        std::vector<Landmark::Ptr> best_kf_landmarks;
        {
            std::lock_guard<std::mutex> lock(best_candidate.kf->mutex_);
            best_kf_landmarks = best_candidate.kf->landmarks_;
        }

        // Lock current_frame_->mutex_ around assign + inner writes to avoid
        // racing with onBACompleted snapshot on the LocalMapping thread.
        {
            std::lock_guard<std::mutex> lock(current_frame_->mutex_);
            current_frame_->landmarks_.assign(current_frame_->keypoints_.size(), nullptr);
            for (int idx : best_success.inlier_indices) {
                if (idx < 0 || idx >= static_cast<int>(best_success.match_indices.size())) {
                    continue;
                }
                const int orig_match_idx = best_success.match_indices[idx];
                if (orig_match_idx < 0 || orig_match_idx >= static_cast<int>(best_candidate.matches.size())) {
                    continue;
                }
                const int kf_idx = best_candidate.matches[orig_match_idx].trainIdx;
                const int curr_idx = best_candidate.matches[orig_match_idx].queryIdx;
                if (kf_idx >= 0 && kf_idx < static_cast<int>(best_kf_landmarks.size()) &&
                    curr_idx >= 0 && curr_idx < static_cast<int>(current_frame_->landmarks_.size())) {
                    current_frame_->landmarks_[curr_idx] = best_kf_landmarks[kf_idx];
                }
            }
        }

        setReferenceKeyframe(best_candidate.kf);

        std::cout << "Relocalize: Matched with KF " << best_candidate.kf->id_
                  << " inliers=" << best_success.inlier_count
                  << " valid_3d=" << best_candidate.valid_3d_matches
                  << " reproj_px=" << best_success.avg_reprojection_error_px
                  << " reproj_valid=" << best_success.valid_reprojection_count
                  << " pose_trans=" << best_success.pose_change_translation
                  << " pose_rot=" << best_success.pose_change_rotation
                  << " dist_to_anchor=" << best_candidate.distance_to_anchor
                  << std::endl;
        return true;
    }

    std::cout << "Relocalize: All candidate PnP attempts failed" << std::endl;
    return false;
}

bool Tracking::reinitialize() {
    if (!current_frame_) return false;

    if (!reinitialization_state_.reference_frame) {
        reinitialization_state_.reference_frame = current_frame_;
        reinitialization_state_.initializer = std::make_shared<Initializer>(current_frame_);
        std::cout << "Tracking: Re-init reference set (frame " << current_frame_->id_ << ")" << std::endl;
        return false;
    }

    if (current_frame_->id_ - reinitialization_state_.reference_frame->id_ < 3) {
        return false;
    }

    const bool ok = reinitialization_state_.initializer->initialize(current_frame_);
    if (!ok) {
        if (current_frame_->id_ - reinitialization_state_.reference_frame->id_ > 30) {
            std::cout << "Tracking: Re-init timeout, resetting reference frame" << std::endl;
            reinitialization_state_.reference_frame = current_frame_;
            reinitialization_state_.initializer = std::make_shared<Initializer>(current_frame_);
        }
        return false;
    }

    std::cout << "Tracking: Re-init triangulation starting..." << std::endl;

    auto kf_ref = std::make_shared<Keyframe>(reinitialization_state_.reference_frame);
    setKeyframeGravity(kf_ref);
    auto kf_cur = std::make_shared<Keyframe>(current_frame_);
    setKeyframeGravity(kf_cur);

    // Set poses - anchor the new segment to the last known good pose
    // This keeps the new segment in the same coordinate system as the existing map
    SE3 T_anchor = recovery_state_.last_good_pose;
    reinitialization_state_.reference_frame->setPose(T_anchor);
    kf_ref->T_cw_ = T_anchor;

    // Poses will be set after scale normalization below

    // Triangulate points in local frame first, then normalize scale, then transform to world
    struct TriPoint {
        Vec3 pos_local;
        int idx_ref;
        int idx_cur;
    };
    std::vector<TriPoint> tri_points;
    std::vector<double> depths;

    for (size_t i = 0; i < reinitialization_state_.initializer->is_triangulated_.size(); ++i) {
        if (!reinitialization_state_.initializer->is_triangulated_[i]) continue;

        cv::Point3f pt3d = reinitialization_state_.initializer->triangulated_points_[i];
        if (!std::isfinite(pt3d.x) || !std::isfinite(pt3d.y) || !std::isfinite(pt3d.z)) continue;
        if (pt3d.z <= 0.0f) continue;
        if (std::abs(pt3d.x) > 1e4f || std::abs(pt3d.y) > 1e4f || std::abs(pt3d.z) > 1e4f) continue;

        depths.push_back(pt3d.z);
        tri_points.push_back({
            Vec3(pt3d.x, pt3d.y, pt3d.z),
            reinitialization_state_.initializer->matches_[i].queryIdx,
            reinitialization_state_.initializer->matches_[i].trainIdx
        });
    }

    if (tri_points.size() < 50) {
        std::cout << "Tracking: Re-init failed - only " << tri_points.size() << " points triangulated" << std::endl;
        reinitialization_state_.reference_frame = current_frame_;
        reinitialization_state_.initializer = std::make_shared<Initializer>(current_frame_);
        return false;
    }

    // Normalize scale in local frame using median depth
    // This sets median depth to 1.0m (reasonable for indoor scenes)
    double scale = 1.0;
    if (!depths.empty()) {
        std::sort(depths.begin(), depths.end());
        double median_depth = depths[depths.size() / 2];
        if (median_depth > 0.0) {
            scale = 1.0 / median_depth;
        }
    }

    // Scale the relative pose T_c1_c2 in local frame, then compose with anchor
    SE3 T_c1_c2_scaled = SE3(
        reinitialization_state_.initializer->T_c1_c2_.so3(),
        reinitialization_state_.initializer->T_c1_c2_.translation() * scale);
    SE3 T_cur = T_c1_c2_scaled * T_anchor;
    kf_cur->T_cw_ = T_cur;
    current_frame_->setPose(T_cur);

    // Transform scaled points from local frame to world frame
    SE3 T_wc_anchor = T_anchor.inverse();
    size_t inserted = 0;

    for (auto& tp : tri_points) {
        Vec3 pos_local_scaled = tp.pos_local * scale;
        Vec3 pos_w = T_wc_anchor * pos_local_scaled;

        static unsigned long reinit_lm_id = 100000;
        auto lm = std::make_shared<Landmark>(reinit_lm_id++, pos_w);

        lm->addObservation(kf_ref, tp.idx_ref);
        lm->addObservation(kf_cur, tp.idx_cur);
        lm->descriptor_ = reinitialization_state_.reference_frame->descriptors_.row(tp.idx_ref).clone();

        kf_ref->landmarks_[tp.idx_ref] = lm;
        kf_cur->landmarks_[tp.idx_cur] = lm;
        {
            std::lock_guard<std::mutex> lock(reinitialization_state_.reference_frame->mutex_);
            reinitialization_state_.reference_frame->landmarks_[tp.idx_ref] = lm;
        }
        {
            std::lock_guard<std::mutex> lock(current_frame_->mutex_);
            current_frame_->landmarks_[tp.idx_cur] = lm;
        }

        if (map_) {
            map_->addLandmark(lm);
            inserted++;
        }
    }

    // Add keyframes to map
    if (map_) {
        map_->addKeyframe(kf_ref);
        map_->addKeyframe(kf_cur);
    }

    setReferenceKeyframe(kf_cur);

    // Reset re-init state
    reinitialization_state_.reference_frame.reset();
    reinitialization_state_.initializer.reset();
    frames_since_successful_relocalization_ = std::numeric_limits<int>::max();

    std::cout << "Tracking: Re-init complete! " << inserted << " landmarks, "
              << map_->getAllKeyframes().size() << " total KFs" << std::endl;
    return true;
}

void Tracking::setKeyframeGravity(Keyframe::Ptr kf) {
    if (!kf || !gravity_aligned_ || accel_buffer_.empty()) return;

    // Find accelerometer readings near keyframe timestamp (±50ms window)
    double ts = kf->timestamp_;
    double window = 0.05;
    std::vector<AccelEntry> nearby;
    for (const auto& a : accel_buffer_) {
        if (a.timestamp_sec >= ts - window && a.timestamp_sec <= ts + window) {
            nearby.push_back(a);
        }
        if (a.timestamp_sec > ts + window) break;
    }

    if (nearby.size() < 3) return;
    // Only trust gravity when the short window is close to stationary.
    // The TUM accel stream is noisy and only approximately camera-aligned.
    if (!AccelerometerProcessor::isStationary(nearby, 1.0)) return;

    // Compute gravity direction in sensor frame, then transform to camera frame
    Vec3 g_sensor = AccelerometerProcessor::estimateGravity(nearby);
    if (g_sensor.norm() < 0.5) return;

    // For TUM datasets, accelerometer frame ≈ camera frame (close enough for prior)
    kf->gravity_in_camera_ = g_sensor.normalized();
    kf->has_gravity_ = true;
}

}
