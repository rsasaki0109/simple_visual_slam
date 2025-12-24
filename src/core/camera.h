#pragma once

#include "core/common.h"

namespace svslam {

class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Camera>;

    Camera(double fx, double fy, double cx, double cy, 
           double k1=0, double k2=0, double p1=0, double p2=0, double k3=0);

    // Project 3D point in camera frame to pixel coordinates
    Vec2 project(const Vec3& p_c) const;

    // Unproject pixel to normalized camera coordinates (z=1)
    Vec3 unproject(const Vec2& p_uv) const;

    // Get Camera Matrix
    cv::Mat K() const;

    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_, k3_;
};

}
