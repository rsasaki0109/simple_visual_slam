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
