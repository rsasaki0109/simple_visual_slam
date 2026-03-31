#include "backend/optimizer.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <set>
#include <algorithm>

namespace svslam {

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
        Eigen::Vector3d t = kf->T_cw_.translation();
        Eigen::Quaterniond q = kf->T_cw_.unit_quaternion();

        param[0] = t.x();
        param[1] = t.y();
        param[2] = t.z();
        // Ceres QuaternionRotatePoint expects [w, x, y, z] order
        param[3] = q.w();
        param[4] = q.x();
        param[5] = q.y();
        param[6] = q.z();
        
        pose_params[kf->id_] = param;

        // Construct Manifold for SE3 (Euclidean<3> x Quaternion)
        // Using QuaternionManifold because we store quaternion in [w, x, y, z] order (Ceres convention)
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
        for (const auto& obs : lm->observations_) {
            auto kf = obs.first.lock();
            if (!kf) continue;
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
        if (lm->isBad()) continue;

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
        
        // Add Residuals
        // Iterate observations
        for (const auto& obs : lm->observations_) {
            auto kf = obs.first.lock();
            if (!kf) continue;
            
            if (pose_params.find(kf->id_) == pose_params.end()) {
                continue; 
            }
            
            const size_t kp_idx = obs.second;
            if (kp_idx >= kf->keypoints_.size()) continue;
            const auto& kp = kf->keypoints_[kp_idx];
            
            auto camera = kf->camera_;
            
            ceres::CostFunction* cost_function = ReprojectionError::Create(
                kp.pt.x, kp.pt.y,
                camera->fx_, camera->fy_, camera->cx_, camera->cy_);
                
            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), pose_params[kf->id_], point_params[lm->id_]);
            ++residual_count;
        }
    }

    // Add depth prior residuals for keyframes with depth images
    int depth_residual_count = 0;
    for (auto& lm : landmarks) {
        if (lm->isBad()) continue;
        if (!point_params.count(lm->id_)) continue;

        for (const auto& obs : lm->observations_) {
            auto kf = obs.first.lock();
            if (!kf) continue;
            if (pose_params.find(kf->id_) == pose_params.end()) continue;
            if (kf->depth_image_.empty()) continue;

            const size_t kp_idx = obs.second;
            if (kp_idx >= kf->keypoints_.size()) continue;
            const auto& kp = kf->keypoints_[kp_idx];

            float depth = kf->getDepth(kp.pt.x, kp.pt.y);
            if (depth <= 0.0f || depth > 10.0f) continue;

            // Weight: higher for sensor depth (metric), lower for DL depth
            double sigma = kf->depth_is_metric_ ? 0.02 : 0.2;
            double weight = 1.0 / sigma;

            auto camera = kf->camera_;
            ceres::CostFunction* depth_cost = DepthPriorError::Create(
                static_cast<double>(depth),
                camera->fx_, camera->fy_, camera->cx_, camera->cy_,
                kp.pt.x, kp.pt.y, weight);

            problem.AddResidualBlock(depth_cost, new ceres::HuberLoss(0.5),
                                     pose_params[kf->id_], point_params[lm->id_]);
            depth_residual_count++;
        }
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

        // Weight: moderate — gravity measurement is noisy but constrains 2 DOF
        double gravity_weight = 5.0;
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

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = iterations;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    // Update State
    for (auto& kf : keyframes) {
        double* param = pose_params[kf->id_];
        Eigen::Vector3d t(param[0], param[1], param[2]);
        // param order is [t, w, x, y, z], Eigen Quaterniond constructor is (w, x, y, z)
        Eigen::Quaterniond q(param[3], param[4], param[5], param[6]); // w, x, y, z
        
        kf->T_cw_ = SE3(q, t);
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
                                      int iterations) {
    if (!map) return;

    // Snapshot keyframes to avoid concurrent modification from Tracking thread
    std::map<unsigned long, Keyframe::Ptr> all_keyframes;
    {
        const auto& kf_ref = map->getAllKeyframes();
        all_keyframes = kf_ref;  // copy
    }
    if (all_keyframes.size() < 2) return;

    ceres::Problem problem;
    std::map<unsigned long, double*> pose_params;
    std::map<unsigned long, Sim3> old_poses;

    auto addPoseParameterBlock = [&](const Keyframe::Ptr& kf, bool constant) {
        if (!kf || pose_params.count(kf->id_)) return;

        double* param = new double[8];
        const Eigen::Vector3d t = kf->T_cw_.translation();
        const Eigen::Quaterniond q = kf->T_cw_.unit_quaternion();
        param[0] = t.x();
        param[1] = t.y();
        param[2] = t.z();
        param[3] = q.w();
        param[4] = q.x();
        param[5] = q.y();
        param[6] = q.z();
        param[7] = 0.0;

        pose_params[kf->id_] = param;
        old_poses[kf->id_] = Sim3(1.0, q, t);

        ceres::Manifold* manifold =
            new ceres::ProductManifold<
                ceres::EuclideanManifold<3>,
                ceres::QuaternionManifold,
                ceres::EuclideanManifold<1>>();
        problem.AddParameterBlock(param, 8, manifold);
        if (constant) {
            problem.SetParameterBlockConstant(param);
        }
    };

    bool first = true;
    for (const auto& kv : all_keyframes) {
        addPoseParameterBlock(kv.second, first);
        first = false;
    }

    auto addEdgeResidual = [&](const PoseGraphEdge& edge, ceres::LossFunction* loss) {
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

    std::set<std::pair<unsigned long, unsigned long>> added_pairs;
    int covisibility_edges = 0;
    for (const auto& kv : all_keyframes) {
        const auto& kf = kv.second;
        if (!kf) continue;
        for (const auto& connected : kf->connected_keyframes_) {
            const auto& other = connected.first;
            if (!other) continue;

            const unsigned long id0 = std::min(kf->id_, other->id_);
            const unsigned long id1 = std::max(kf->id_, other->id_);
            if (!added_pairs.insert({id0, id1}).second) continue;

            const double weight_scale =
                std::clamp(std::sqrt(static_cast<double>(connected.second) / 30.0), 0.75, 2.0);
            const Eigen::Quaterniond q_from = kf->T_cw_.unit_quaternion();
            const Eigen::Quaterniond q_to = other->T_cw_.unit_quaternion();
            PoseGraphEdge edge;
            edge.from = kf;
            edge.to = other;
            edge.relative_pose =
                Sim3(1.0, q_to, other->T_cw_.translation()) *
                Sim3(1.0, q_from, kf->T_cw_.translation()).inverse();
            edge.translation_weight = weight_scale;
            edge.rotation_weight = weight_scale;
            edge.scale_weight = 1.5 * weight_scale;
            addEdgeResidual(edge, new ceres::HuberLoss(1.0));
            ++covisibility_edges;
        }
    }

    int loop_edge_count = 0;
    for (const auto& edge : loop_edges) {
        addEdgeResidual(edge, new ceres::HuberLoss(0.5));
        ++loop_edge_count;
    }

    if (covisibility_edges == 0 && loop_edge_count == 0) {
        for (auto& kv : pose_params) {
            delete[] kv.second;
        }
        return;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = iterations;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "PoseGraph: " << summary.BriefReport()
              << " | covisibility_edges=" << covisibility_edges
              << " | loop_edges=" << loop_edge_count << std::endl;

    std::map<unsigned long, Sim3> optimized_poses;
    for (const auto& kv : all_keyframes) {
        auto* param = pose_params[kv.first];
        Eigen::Vector3d t(param[0], param[1], param[2]);
        Eigen::Quaterniond q(param[3], param[4], param[5], param[6]);
        const double scale = std::exp(param[7]);

        // Validate optimized pose
        if (!std::isfinite(t.x()) || !std::isfinite(t.y()) || !std::isfinite(t.z()) ||
            !std::isfinite(q.w()) || !std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
            !std::isfinite(scale) || scale < 0.01 || scale > 100.0) {
            // Keep original pose
            optimized_poses[kv.first] = old_poses[kv.first];
            continue;
        }

        q.normalize();
        optimized_poses[kv.first] = Sim3(scale, q, t);
        kv.second->T_cw_ = SE3(q, t / scale);
    }

    // Snapshot landmarks to avoid concurrent modification
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
                return a->observations_.size() > b->observations_.size();
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
