#include "backend/optimizer.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <set>
#include <vector>

namespace svslam {

namespace {

// Default 1 for repeatable BA; use SVSLAM_CERES_NUM_THREADS=N (integer >= 1) for faster solves.
int ceres_num_threads_from_env() {
    const char* env = std::getenv("SVSLAM_CERES_NUM_THREADS");
    if (!env || env[0] == '\0') {
        return 1;
    }
    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env || v < 1) {
        return 1;
    }
    if (v > 64) {
        return 64;
    }
    return static_cast<int>(v);
}

struct ObservationItem {
    unsigned long keyframe_id = 0;
    std::size_t keypoint_index = 0;
    Keyframe::Ptr keyframe;
};

void fillSe3ParameterBlock(const SE3& pose, double* param) {
    const Eigen::Vector3d translation = pose.translation();
    const Eigen::Quaterniond quaternion = pose.unit_quaternion();
    param[0] = translation.x();
    param[1] = translation.y();
    param[2] = translation.z();
    param[3] = quaternion.w();
    param[4] = quaternion.x();
    param[5] = quaternion.y();
    param[6] = quaternion.z();
}

SE3 se3FromParameterBlock(const double* param) {
    const Eigen::Vector3d translation(param[0], param[1], param[2]);
    const Eigen::Quaterniond quaternion(param[3], param[4], param[5], param[6]);
    return SE3(quaternion, translation);
}

Sim3 sim3FromPoseGraphParameterBlock(const double* param) {
    const Eigen::Vector3d translation(param[0], param[1], param[2]);
    Eigen::Quaterniond quaternion(param[3], param[4], param[5], param[6]);
    quaternion.normalize();
    return Sim3(std::exp(param[7]), quaternion, translation);
}

std::vector<ObservationItem> collectSortedObservations(const Landmark::Ptr& landmark) {
    std::vector<ObservationItem> observations;
    if (!landmark) {
        return observations;
    }

    std::lock_guard<std::mutex> observation_lock(landmark->mutex_);
    observations.reserve(landmark->observations_.size());
    for (const auto& observation : landmark->observations_) {
        auto keyframe = observation.first.lock();
        if (!keyframe) {
            continue;
        }
        observations.push_back({keyframe->id_, observation.second, keyframe});
    }

    std::sort(observations.begin(), observations.end(),
              [](const ObservationItem& lhs, const ObservationItem& rhs) {
                  if (lhs.keyframe_id != rhs.keyframe_id) {
                      return lhs.keyframe_id < rhs.keyframe_id;
                  }
                  return lhs.keypoint_index < rhs.keypoint_index;
              });
    return observations;
}

}  // namespace

// Define Cost Functions here

// Reprojection Error Cost Function
struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera_pose, // [tx, ty, tz, qw, qx, qy, qz]
                    const T* const point,       // [x, y, z]
                    T* residuals) const {
        
        // Transform point from world to camera: P_c = R * P_w + t
        // camera_pose[0-2] is t
        // camera_pose[3-6] is q (x, y, z, w)
        
        T p[3];
        ceres::QuaternionRotatePoint(camera_pose + 3, point, p);
        p[0] += camera_pose[0];
        p[1] += camera_pose[1];
        p[2] += camera_pose[2];
        
        // Projection
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);
        
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const double fx,
                                       const double fy,
                                       const double cx,
                                       const double cy) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
            new ReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
    }
    
    double observed_x;
    double observed_y;
    double fx, fy, cx, cy;
};

namespace {

void addReprojectionResiduals(ceres::Problem& problem,
                              const Landmark::Ptr& landmark,
                              double* point_param,
                              const std::map<unsigned long, double*>& pose_params,
                              int& residual_count) {
    for (const auto& observation : collectSortedObservations(landmark)) {
        const auto pose_it = pose_params.find(observation.keyframe_id);
        if (pose_it == pose_params.end()) {
            continue;
        }
        if (observation.keypoint_index >= observation.keyframe->keypoints_.size()) {
            continue;
        }

        const auto& keypoint = observation.keyframe->keypoints_[observation.keypoint_index];
        const auto& camera = observation.keyframe->camera_;
        ceres::CostFunction* cost_function = ReprojectionError::Create(
            keypoint.pt.x, keypoint.pt.y,
            camera->fx_, camera->fy_, camera->cx_, camera->cy_);
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0),
                                 pose_it->second, point_param);
        ++residual_count;
    }
}

int addDepthPriorResiduals(ceres::Problem& problem,
                           const Landmark::Ptr& landmark,
                           double* point_param,
                           const std::map<unsigned long, double*>& pose_params) {
    int depth_residual_count = 0;
    for (const auto& observation : collectSortedObservations(landmark)) {
        const auto pose_it = pose_params.find(observation.keyframe_id);
        if (pose_it == pose_params.end()) {
            continue;
        }
        const auto& keyframe = observation.keyframe;
        if (keyframe->depth_image_.empty() ||
            observation.keypoint_index >= keyframe->keypoints_.size()) {
            continue;
        }

        const auto& keypoint = keyframe->keypoints_[observation.keypoint_index];
        const float depth = keyframe->getDepth(keypoint.pt.x, keypoint.pt.y);
        if (depth <= 0.0f || depth > 10.0f) {
            continue;
        }

        const double sigma = keyframe->depth_is_metric_ ? 0.015 : 0.2;
        const double weight = 1.0 / sigma;
        const auto& camera = keyframe->camera_;
        ceres::CostFunction* depth_cost = DepthPriorError::Create(
            static_cast<double>(depth),
            camera->fx_, camera->fy_, camera->cx_, camera->cy_,
            keypoint.pt.x, keypoint.pt.y, weight);
        problem.AddResidualBlock(depth_cost, new ceres::HuberLoss(0.5),
                                 pose_it->second, point_param);
        ++depth_residual_count;
    }
    return depth_residual_count;
}

struct PoseGraphKeyframeSnapshot {
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    std::vector<std::pair<Keyframe::Ptr, int>> connections;
};

std::map<unsigned long, PoseGraphKeyframeSnapshot> snapshotPoseGraphKeyframes(
    const std::map<unsigned long, Keyframe::Ptr>& all_keyframes) {
    std::map<unsigned long, PoseGraphKeyframeSnapshot> snapshots;
    for (const auto& entry : all_keyframes) {
        const auto& keyframe = entry.second;
        if (!keyframe) {
            continue;
        }

        PoseGraphKeyframeSnapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(keyframe->mutex_);
            snapshot.translation = keyframe->T_cw_.translation();
            snapshot.rotation = keyframe->T_cw_.unit_quaternion();
            snapshot.connections.reserve(keyframe->connected_keyframes_.size());
            for (const auto& connection : keyframe->connected_keyframes_) {
                snapshot.connections.push_back(connection);
            }
        }
        snapshots.emplace(entry.first, std::move(snapshot));
    }
    return snapshots;
}

void addPoseGraphParameterBlock(ceres::Problem& problem,
                                const Keyframe::Ptr& keyframe,
                                const std::map<unsigned long, PoseGraphKeyframeSnapshot>& snapshots,
                                std::map<unsigned long, double*>& pose_params,
                                std::map<unsigned long, Sim3>& old_poses,
                                bool constant) {
    if (!keyframe || pose_params.count(keyframe->id_)) {
        return;
    }
    const auto snapshot_it = snapshots.find(keyframe->id_);
    if (snapshot_it == snapshots.end()) {
        return;
    }

    double* param = new double[8];
    fillSe3ParameterBlock(SE3(snapshot_it->second.rotation, snapshot_it->second.translation), param);
    param[7] = 0.0;
    pose_params[keyframe->id_] = param;
    old_poses[keyframe->id_] = Sim3(1.0, snapshot_it->second.rotation, snapshot_it->second.translation);

    ceres::Manifold* manifold =
        new ceres::ProductManifold<
            ceres::EuclideanManifold<3>,
            ceres::QuaternionManifold,
            ceres::EuclideanManifold<1>>();
    problem.AddParameterBlock(param, 8, manifold);
    if (constant) {
        problem.SetParameterBlockConstant(param);
    }
}

void attachPoseGraphParameterBlock(ceres::Problem& problem,
                                   double* param,
                                   bool constant) {
    ceres::Manifold* manifold =
        new ceres::ProductManifold<
            ceres::EuclideanManifold<3>,
            ceres::QuaternionManifold,
            ceres::EuclideanManifold<1>>();
    problem.AddParameterBlock(param, 8, manifold);
    if (constant) {
        problem.SetParameterBlockConstant(param);
    }
}

}  // namespace

struct PoseGraphError {
    PoseGraphError(const Sim3& measurement,
                   double translation_weight,
                   double rotation_weight,
                   double scale_weight)
        : translation_weight_(translation_weight),
          rotation_weight_(rotation_weight),
          scale_weight_(scale_weight) {
        t_meas_[0] = measurement.translation().x();
        t_meas_[1] = measurement.translation().y();
        t_meas_[2] = measurement.translation().z();

        const Eigen::Quaterniond q(measurement.rotationMatrix());
        q_meas_[0] = q.w();
        q_meas_[1] = q.x();
        q_meas_[2] = q.y();
        q_meas_[3] = q.z();
        log_scale_meas_ = std::log(measurement.scale());
    }

    template <typename T>
    bool operator()(const T* const pose_from,
                    const T* const pose_to,
                    T* residuals) const {
        const T q_from[4] = {pose_from[3], pose_from[4], pose_from[5], pose_from[6]};
        const T q_to[4] = {pose_to[3], pose_to[4], pose_to[5], pose_to[6]};
        const T scale_from = ceres::exp(pose_from[7]);
        const T scale_to = ceres::exp(pose_to[7]);
        const T scale_rel = scale_to / scale_from;

        const T q_from_inv[4] = {q_from[0], -q_from[1], -q_from[2], -q_from[3]};

        T q_pred[4];
        ceres::QuaternionProduct(q_to, q_from_inv, q_pred);

        const T t_from[3] = {pose_from[0], pose_from[1], pose_from[2]};
        const T t_to[3] = {pose_to[0], pose_to[1], pose_to[2]};

        T rotated_t_from[3];
        ceres::QuaternionRotatePoint(q_pred, t_from, rotated_t_from);

        const T t_pred[3] = {
            t_to[0] - scale_rel * rotated_t_from[0],
            t_to[1] - scale_rel * rotated_t_from[1],
            t_to[2] - scale_rel * rotated_t_from[2]
        };

        residuals[0] = T(translation_weight_) * (t_pred[0] - T(t_meas_[0]));
        residuals[1] = T(translation_weight_) * (t_pred[1] - T(t_meas_[1]));
        residuals[2] = T(translation_weight_) * (t_pred[2] - T(t_meas_[2]));

        const T q_meas_inv[4] = {
            T(q_meas_[0]),
            T(-q_meas_[1]),
            T(-q_meas_[2]),
            T(-q_meas_[3])
        };
        T q_err[4];
        ceres::QuaternionProduct(q_meas_inv, q_pred, q_err);
        residuals[3] = T(rotation_weight_) * T(2.0) * q_err[1];
        residuals[4] = T(rotation_weight_) * T(2.0) * q_err[2];
        residuals[5] = T(rotation_weight_) * T(2.0) * q_err[3];
        residuals[6] = T(scale_weight_) * ((pose_to[7] - pose_from[7]) - T(log_scale_meas_));
        return true;
    }

    static ceres::CostFunction* Create(const Sim3& measurement,
                                       double translation_weight,
                                       double rotation_weight,
                                       double scale_weight) {
        return new ceres::AutoDiffCostFunction<PoseGraphError, 7, 8, 8>(
            new PoseGraphError(measurement, translation_weight, rotation_weight, scale_weight));
    }

    double t_meas_[3];
    double q_meas_[4];
    double log_scale_meas_;
    double translation_weight_;
    double rotation_weight_;
    double scale_weight_;
};

struct ScalePriorError {
    explicit ScalePriorError(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        residuals[0] = T(weight_) * pose[7];
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<ScalePriorError, 1, 8>(
            new ScalePriorError(weight));
    }

    double weight_;
};

void Optimizer::bundleAdjustment(const std::vector<Keyframe::Ptr>& keyframes, 
                                 const std::vector<Landmark::Ptr>& landmarks,
                                 int iterations) {
    ceres::Problem problem;
    std::set<unsigned long> local_keyframe_ids;
    
    // Parameter blocks
    // 1. Keyframe Poses
    // We use double[7] for T_cw (translation + quaternion)
    // Map: KF ID -> double*
    std::map<unsigned long, double*> pose_params;

    auto addPoseParameterBlock = [&](const Keyframe::Ptr& kf, bool constant) {
        if (!kf || pose_params.count(kf->id_)) return;

        double* param = new double[7];
        fillSe3ParameterBlock(kf->T_cw_, param);
        pose_params[kf->id_] = param;

        ceres::Manifold* manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::QuaternionManifold>();

        problem.AddParameterBlock(param, 7, manifold);

        if (constant) {
            problem.SetParameterBlockConstant(param);
        }
    };

    for (auto& kf : keyframes) {
        if (!kf) continue;
        local_keyframe_ids.insert(kf->id_);
        addPoseParameterBlock(kf, false);
    }

    std::map<unsigned long, Keyframe::Ptr> fixed_keyframes;
    for (auto& lm : landmarks) {
        if (!lm || lm->isBad()) continue;
        for (const auto& observation : collectSortedObservations(lm)) {
            const auto& kf = observation.keyframe;
            if (local_keyframe_ids.count(kf->id_)) continue;
            fixed_keyframes[kf->id_] = kf;
        }
    }

    for (const auto& kv : fixed_keyframes) {
        addPoseParameterBlock(kv.second, true);
    }

    if (fixed_keyframes.empty() && !keyframes.empty()) {
        problem.SetParameterBlockConstant(pose_params[keyframes.front()->id_]);
    }

    // 2. Landmarks
    std::map<unsigned long, double*> point_params;
    int residual_count = 0;

    // Bounds for landmark positions to prevent divergence
    const double position_bound = 100.0; // Reasonable bound for indoor scenes

    for (auto& lm : landmarks) {
        if (!lm || lm->isBad()) continue;

        Vec3 pos = lm->getPos();

        // Skip landmarks with already invalid positions
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
            lm->setBad();
            continue;
        }
        if (std::abs(pos.x()) > position_bound || std::abs(pos.y()) > position_bound || std::abs(pos.z()) > position_bound) {
            lm->setBad();
            continue;
        }

        double* param = new double[3];
        param[0] = pos.x();
        param[1] = pos.y();
        param[2] = pos.z();

        point_params[lm->id_] = param;

        problem.AddParameterBlock(param, 3);

        // Add bounds to prevent divergence
        problem.SetParameterLowerBound(param, 0, -position_bound);
        problem.SetParameterUpperBound(param, 0, position_bound);
        problem.SetParameterLowerBound(param, 1, -position_bound);
        problem.SetParameterUpperBound(param, 1, position_bound);
        problem.SetParameterLowerBound(param, 2, -position_bound);
        problem.SetParameterUpperBound(param, 2, position_bound);
        
        addReprojectionResiduals(problem, lm, param, pose_params, residual_count);
    }

    int depth_residual_count = 0;
    for (auto& lm : landmarks) {
        if (!lm || lm->isBad()) continue;
        if (!point_params.count(lm->id_)) continue;
        depth_residual_count += addDepthPriorResiduals(
            problem, lm, point_params[lm->id_], pose_params);
    }

    if (depth_residual_count > 0) {
        std::cout << "BA: Added " << depth_residual_count << " depth prior residuals" << std::endl;
    }

    // Add gravity prior residuals for keyframes with accelerometer data
    int gravity_residual_count = 0;
    for (auto& kf : keyframes) {
        if (!kf || !kf->has_gravity_) continue;
        if (pose_params.find(kf->id_) == pose_params.end()) continue;
        if (problem.IsParameterBlockConstant(pose_params[kf->id_])) continue;

        // Gravity is only approximately camera-aligned on TUM, so keep this prior soft.
        double gravity_weight = 2.0;
        ceres::CostFunction* gravity_cost = GravityPriorError::Create(
            kf->gravity_in_camera_.x(), kf->gravity_in_camera_.y(), kf->gravity_in_camera_.z(),
            gravity_weight);

        problem.AddResidualBlock(gravity_cost, new ceres::HuberLoss(0.3),
                                 pose_params[kf->id_]);
        gravity_residual_count++;
    }

    if (gravity_residual_count > 0) {
        std::cout << "BA: Added " << gravity_residual_count << " gravity prior residuals" << std::endl;
    }

    if (residual_count == 0 && depth_residual_count == 0) {
        for (auto& kv : pose_params) {
            delete[] kv.second;
        }
        for (auto& kv : point_params) {
            delete[] kv.second;
        }
        return;
    }

    // Solve: default single-thread for repeatable BA; override via SVSLAM_CERES_NUM_THREADS.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = ceres_num_threads_from_env();
    options.max_num_iterations = iterations;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    // Update State
    for (auto& kf : keyframes) {
        kf->T_cw_ = se3FromParameterBlock(pose_params[kf->id_]);
    }
    
    for (auto& lm : landmarks) {
        if (point_params.count(lm->id_)) {
            double* param = point_params[lm->id_];
            Vec3 pos(param[0], param[1], param[2]);

            // Validate updated position before applying
            bool valid = std::isfinite(pos.x()) && std::isfinite(pos.y()) && std::isfinite(pos.z());
            valid = valid && (std::abs(pos.x()) < position_bound && std::abs(pos.y()) < position_bound && std::abs(pos.z()) < position_bound);

            if (valid) {
                lm->setPos(pos);
            } else {
                lm->setBad();
            }
        }
    }

    for (auto& kv : pose_params) {
        delete[] kv.second;
    }
    for (auto& kv : point_params) {
        delete[] kv.second;
    }
}

void Optimizer::poseGraphOptimization(Map::Ptr map,
                                      const std::vector<PoseGraphEdge>& loop_edges,
                                      int iterations,
                                      bool fix_scale) {
    if (!map) return;

    std::map<unsigned long, Keyframe::Ptr> all_keyframes;
    {
        const auto& kf_ref = map->getAllKeyframes();
        all_keyframes = kf_ref;
    }
    if (all_keyframes.size() < 2) return;

    const auto keyframe_snapshots = snapshotPoseGraphKeyframes(all_keyframes);

    ceres::Problem bootstrap_problem;
    std::map<unsigned long, double*> pose_params;
    std::map<unsigned long, Sim3> old_poses;
    std::set<unsigned long> constant_pose_ids;

    bool first = true;
    for (const auto& kv : all_keyframes) {
        addPoseGraphParameterBlock(bootstrap_problem, kv.second, keyframe_snapshots,
                                   pose_params, old_poses, first);
        if (first) {
            constant_pose_ids.insert(kv.first);
        }
        first = false;
    }

    auto addEdgeResidual = [&](ceres::Problem& problem,
                               const PoseGraphEdge& edge,
                               ceres::LossFunction* loss) {
        if (!edge.from || !edge.to) return;
        auto from_it = pose_params.find(edge.from->id_);
        auto to_it = pose_params.find(edge.to->id_);
        if (from_it == pose_params.end() || to_it == pose_params.end()) return;
        ceres::CostFunction* cost = PoseGraphError::Create(
            edge.relative_pose,
            edge.translation_weight,
            edge.rotation_weight,
            edge.scale_weight);
        problem.AddResidualBlock(cost, loss, from_it->second, to_it->second);
    };

    auto poseGraphResidualNorm = [&](const PoseGraphEdge& edge) {
        if (!edge.from || !edge.to) {
            return 0.0;
        }
        const auto from_it = pose_params.find(edge.from->id_);
        const auto to_it = pose_params.find(edge.to->id_);
        if (from_it == pose_params.end() || to_it == pose_params.end()) {
            return 0.0;
        }
        PoseGraphError error(edge.relative_pose,
                             edge.translation_weight,
                             edge.rotation_weight,
                             edge.scale_weight);
        std::array<double, 7> residuals{};
        error(from_it->second, to_it->second, residuals.data());
        double sq_norm = 0.0;
        for (const double residual : residuals) {
            sq_norm += residual * residual;
        }
        return std::sqrt(sq_norm);
    };

    auto sharedLandmarkCount = [](const Keyframe::Ptr& from,
                                  const PoseGraphKeyframeSnapshot& other_snapshot,
                                  int forward_count) {
        int shared_landmarks = std::max(forward_count, 0);
        if (!from) {
            return shared_landmarks;
        }
        for (const auto& reverse_connection : other_snapshot.connections) {
            if (!reverse_connection.first || reverse_connection.first->id_ != from->id_) {
                continue;
            }
            shared_landmarks = std::max(shared_landmarks, reverse_connection.second);
            break;
        }
        return shared_landmarks;
    };

    int max_shared_landmarks = 0;
    for (const auto& kv : all_keyframes) {
        const auto& kf = kv.second;
        if (!kf) continue;
        const auto snapshot_it = keyframe_snapshots.find(kf->id_);
        if (snapshot_it == keyframe_snapshots.end()) continue;
        const auto& snapshot = snapshot_it->second;

        for (const auto& connected : snapshot.connections) {
            const auto& other = connected.first;
            if (!other) continue;
            const auto other_snapshot_it = keyframe_snapshots.find(other->id_);
            if (other_snapshot_it == keyframe_snapshots.end()) continue;
            max_shared_landmarks =
                std::max(max_shared_landmarks,
                         sharedLandmarkCount(kf, other_snapshot_it->second, connected.second));
        }
    }

    struct WeightedPoseGraphEdge {
        PoseGraphEdge edge;
        bool is_loop = false;
        double irls_weight = 1.0;
    };

    std::vector<WeightedPoseGraphEdge> weighted_edges;
    weighted_edges.reserve(all_keyframes.size() * 6 + loop_edges.size());

    std::set<std::pair<unsigned long, unsigned long>> added_pairs;
    int covisibility_edges = 0;
    for (const auto& kv : all_keyframes) {
        const auto& kf = kv.second;
        if (!kf) continue;
        const auto snapshot_it = keyframe_snapshots.find(kf->id_);
        if (snapshot_it == keyframe_snapshots.end()) continue;
        const auto& snapshot = snapshot_it->second;

        for (const auto& connected : snapshot.connections) {
            const auto& other = connected.first;
            if (!other) continue;
            const auto other_snapshot_it = keyframe_snapshots.find(other->id_);
            if (other_snapshot_it == keyframe_snapshots.end()) continue;
            const auto& other_snapshot = other_snapshot_it->second;

            const unsigned long id0 = std::min(kf->id_, other->id_);
            const unsigned long id1 = std::max(kf->id_, other->id_);
            if (!added_pairs.insert({id0, id1}).second) continue;

            const int shared_landmarks =
                sharedLandmarkCount(kf, other_snapshot, connected.second);
            const double normalized_shared_landmarks =
                max_shared_landmarks > 0
                    ? static_cast<double>(shared_landmarks) / static_cast<double>(max_shared_landmarks)
                    : 1.0;
            const double weight_scale =
                std::sqrt(std::clamp(normalized_shared_landmarks, 0.0, 1.0));
            PoseGraphEdge edge;
            edge.from = kf;
            edge.to = other;
            edge.relative_pose =
                Sim3(1.0, other_snapshot.rotation, other_snapshot.translation) *
                Sim3(1.0, snapshot.rotation, snapshot.translation).inverse();
            edge.translation_weight = weight_scale;
            edge.rotation_weight = weight_scale;
            edge.scale_weight = fix_scale ? 100.0 : (1.5 * weight_scale);
            weighted_edges.push_back({edge, false, 1.0});
            ++covisibility_edges;
        }
    }

    int loop_edge_count = 0;
    for (const auto& edge : loop_edges) {
        weighted_edges.push_back({edge, true, 1.0});
        ++loop_edge_count;
    }

    if (covisibility_edges == 0 && loop_edge_count == 0) {
        for (auto& kv : pose_params) {
            delete[] kv.second;
        }
        return;
    }

    auto medianOf = [](std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }
        const auto mid = values.begin() + (values.size() / 2);
        std::nth_element(values.begin(), mid, values.end());
        double median = *mid;
        if ((values.size() % 2) == 0) {
            const auto lower_mid = values.begin() + (values.size() / 2 - 1);
            std::nth_element(values.begin(), lower_mid, values.end());
            median = 0.5 * (median + *lower_mid);
        }
        return median;
    };

    auto buildProblem = [&](ceres::Problem& problem) {
        for (const auto& kv : pose_params) {
            attachPoseGraphParameterBlock(problem,
                                          kv.second,
                                          constant_pose_ids.count(kv.first) > 0);
        }

        if (fix_scale) {
            constexpr double kMetricScalePriorWeight = 100.0;
            for (const auto& kv : pose_params) {
                if (constant_pose_ids.count(kv.first) == 0) {
                    problem.AddResidualBlock(
                        ScalePriorError::Create(kMetricScalePriorWeight),
                        nullptr,
                        kv.second);
                }
            }
        }

        for (const auto& weighted_edge : weighted_edges) {
            PoseGraphEdge scaled_edge = weighted_edge.edge;
            const double w = weighted_edge.irls_weight;
            scaled_edge.translation_weight *= w;
            scaled_edge.rotation_weight *= w;
            scaled_edge.scale_weight *= w;
            ceres::LossFunction* loss = weighted_edge.is_loop
                ? static_cast<ceres::LossFunction*>(new ceres::CauchyLoss(1.0))
                : static_cast<ceres::LossFunction*>(new ceres::HuberLoss(1.0));
            addEdgeResidual(problem, scaled_edge, loss);
        }
    };

    auto solvePoseGraph = [&](const char* pass_name) {
        ceres::Problem problem;
        buildProblem(problem);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        options.num_threads = ceres_num_threads_from_env();
        options.max_num_iterations = std::max(iterations, loop_edge_count > 0 ? 90 : iterations);
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;

        std::cout << "PoseGraph(" << pass_name
                  << "): linear_solver=SPARSE_NORMAL_CHOLESKY"
                  << " sparse_library=SUITE_SPARSE"
                  << " loss_loop=Cauchy"
                  << " loss_covisibility=Huber"
                  << " max_iterations=" << options.max_num_iterations
                  << std::endl;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        return summary;
    };

    ceres::Solver::Summary summary = solvePoseGraph("pass1");

    int irls_downweighted_edges = 0;
    double irls_min_weight = 1.0;
    double irls_max_weight = 1.0;
    if (loop_edge_count > 0) {
        std::vector<double> loop_residuals;
        loop_residuals.reserve(loop_edge_count);
        for (const auto& weighted_edge : weighted_edges) {
            if (!weighted_edge.is_loop) {
                continue;
            }
            loop_residuals.push_back(poseGraphResidualNorm(weighted_edge.edge));
        }

        const double residual_median = medianOf(loop_residuals);
        std::vector<double> residual_deviations;
        residual_deviations.reserve(loop_residuals.size());
        for (const double residual : loop_residuals) {
            residual_deviations.push_back(std::abs(residual - residual_median));
        }
        const double residual_mad = medianOf(residual_deviations);
        const double residual_cutoff = std::max(1.0, residual_median + 2.5 * residual_mad);

        bool rerun_irls = false;
        for (auto& weighted_edge : weighted_edges) {
            if (!weighted_edge.is_loop) {
                continue;
            }
            const double residual = poseGraphResidualNorm(weighted_edge.edge);
            double weight = 1.0;
            if (residual > residual_cutoff) {
                weight = std::clamp(std::sqrt(residual_cutoff / residual), 0.20, 1.0);
            }
            weighted_edge.irls_weight = weight;
            irls_min_weight = std::min(irls_min_weight, weight);
            irls_max_weight = std::max(irls_max_weight, weight);
            if (weight < 0.999) {
                rerun_irls = true;
                ++irls_downweighted_edges;
            }
        }

        std::cout << "PoseGraph: IRLS residual_median=" << residual_median
                  << " residual_mad=" << residual_mad
                  << " cutoff=" << residual_cutoff
                  << " downweighted_loop_edges=" << irls_downweighted_edges
                  << " min_loop_weight=" << irls_min_weight
                  << " max_loop_weight=" << irls_max_weight
                  << std::endl;

        if (rerun_irls) {
            summary = solvePoseGraph("pass2_irls");
        }
    }

    double min_scale = std::numeric_limits<double>::infinity();
    double max_scale = 0.0;
    double sum_scale = 0.0;
    std::size_t counted_scales = 0;

    std::cout << "PoseGraph: " << summary.BriefReport()
              << " | covisibility_edges=" << covisibility_edges
              << " | loop_edges=" << loop_edge_count
              << " | irls_downweighted_loop_edges=" << irls_downweighted_edges
              << std::endl;

    std::map<unsigned long, Sim3> optimized_poses;
    for (const auto& kv : all_keyframes) {
        auto* param = pose_params[kv.first];
        const Sim3 optimized_pose = sim3FromPoseGraphParameterBlock(param);
        const Eigen::Vector3d t = optimized_pose.translation();
        const Eigen::Quaterniond q(optimized_pose.rotationMatrix());
        const double scale = optimized_pose.scale();

        if (!std::isfinite(t.x()) || !std::isfinite(t.y()) || !std::isfinite(t.z()) ||
            !std::isfinite(q.w()) || !std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
            !std::isfinite(scale) || scale < 0.01 || scale > 100.0) {
            optimized_poses[kv.first] = old_poses[kv.first];
            continue;
        }

        optimized_poses[kv.first] = optimized_pose;
        {
            std::lock_guard<std::mutex> lock(kv.second->mutex_);
            kv.second->T_cw_ = SE3(q, t / scale);
        }

        min_scale = std::min(min_scale, scale);
        max_scale = std::max(max_scale, scale);
        sum_scale += scale;
        counted_scales++;
    }

    if (counted_scales > 0) {
        std::cout << "PoseGraph: scale_stats min=" << min_scale
                  << " max=" << max_scale
                  << " mean=" << (sum_scale / static_cast<double>(counted_scales))
                  << " | metric_fix_scale=" << (fix_scale ? 1 : 0) << std::endl;
    }

    std::map<unsigned long, Landmark::Ptr> all_landmarks;
    {
        const auto& lm_ref = map->getAllLandmarks();
        all_landmarks = lm_ref;
    }
    for (const auto& kv : all_landmarks) {
        const auto& lm = kv.second;
        if (!lm || lm->isBad()) continue;

        Keyframe::Ptr reference_kf;
        {
            std::unique_lock<std::mutex> lm_lock(lm->mutex_);
            for (const auto& obs : lm->observations_) {
                auto kf = obs.first.lock();
                if (kf && old_poses.count(kf->id_)) {
                    reference_kf = kf;
                    break;
                }
            }
        }
        if (!reference_kf) continue;

        const Sim3 world_delta = optimized_poses.at(reference_kf->id_).inverse() * old_poses.at(reference_kf->id_);
        const Vec3 updated_pos = world_delta * lm->getPos();
        if (std::isfinite(updated_pos.x()) && std::isfinite(updated_pos.y()) && std::isfinite(updated_pos.z())) {
            lm->setPos(updated_pos);
        }
    }

    for (const auto& kv : pose_params) {
        delete[] kv.second;
    }
}

void Optimizer::globalBundleAdjustment(Map::Ptr map, int iterations) {
    if (!map) return;

    const auto& all_keyframes_map = map->getAllKeyframes();
    const auto& all_landmarks_map = map->getAllLandmarks();
    if (all_keyframes_map.size() < 2 || all_landmarks_map.size() < 10) return;

    std::vector<Keyframe::Ptr> keyframes;
    keyframes.reserve(all_keyframes_map.size());
    for (const auto& kv : all_keyframes_map) {
        if (kv.second) keyframes.push_back(kv.second);
    }

    std::vector<Landmark::Ptr> landmarks;
    landmarks.reserve(all_landmarks_map.size());
    for (const auto& kv : all_landmarks_map) {
        if (kv.second && !kv.second->isBad()) landmarks.push_back(kv.second);
    }

    // Limit landmarks to prevent divergence in large maps
    const size_t max_global_ba_landmarks = 2000;
    if (landmarks.size() > max_global_ba_landmarks) {
        // Keep landmarks with most observations (most constrained)
        std::sort(landmarks.begin(), landmarks.end(),
            [](const Landmark::Ptr& a, const Landmark::Ptr& b) {
                const size_t a_obs = a ? a->observations_.size() : 0;
                const size_t b_obs = b ? b->observations_.size() : 0;
                if (a_obs != b_obs) return a_obs > b_obs;
                return a->id_ < b->id_;
            });
        landmarks.resize(max_global_ba_landmarks);
    }

    std::cout << "GlobalBA: start on " << keyframes.size()
              << " KFs and " << landmarks.size() << " LMs." << std::endl;
    bundleAdjustment(keyframes, landmarks, iterations);
}

int Optimizer::poseOptimization(Frame::Ptr frame) {
    // Stub
    return 0;
}

}
