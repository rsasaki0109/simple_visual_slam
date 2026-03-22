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
                                      int iterations = 50);

    static void globalBundleAdjustment(Map::Ptr map, int iterations = 10);
                                 
    // Pose optimization only (e.g. for tracking)
    static int poseOptimization(Frame::Ptr frame);
};

}
