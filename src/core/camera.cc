#include "core/camera.h"

namespace svslam {

Camera::Camera(double fx, double fy, double cx, double cy, 
               double k1, double k2, double p1, double p2, double k3)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
      k1_(k1), k2_(k2), p1_(p1), p2_(p2), k3_(k3) {}

Vec2 Camera::project(const Vec3& p_c) const {
    // Simple pinhole + distortion model
    double x = p_c[0] / p_c[2];
    double y = p_c[1] / p_c[2];

    double r2 = x*x + y*y;
    double r4 = r2*r2;
    double r6 = r2*r4;
    
    double radial = 1.0 + k1_*r2 + k2_*r4 + k3_*r6;
    double tangential_x = 2.0*p1_*x*y + p2_*(r2 + 2.0*x*x);
    double tangential_y = p1_*(r2 + 2.0*y*y) + 2.0*p2_*x*y;

    double x_d = x * radial + tangential_x;
    double y_d = y * radial + tangential_y;

    return Vec2(fx_ * x_d + cx_, fy_ * y_d + cy_);
}

Vec3 Camera::unproject(const Vec2& p_uv) const {
    // Iterative undistortion could be implemented here for better accuracy
    // For now, assuming simple inverse of pinhole if distortion is small or handled elsewhere,
    // or typically we use undistorted images for SLAM.
    // Let's implement basic pinhole unproject for now.
    // Ideally we should undistort the point first.
    
    double x = (p_uv[0] - cx_) / fx_;
    double y = (p_uv[1] - cy_) / fy_;
    return Vec3(x, y, 1.0);
}

cv::Mat Camera::K() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0,0) = fx_;
    K.at<double>(1,1) = fy_;
    K.at<double>(0,2) = cx_;
    K.at<double>(1,2) = cy_;
    return K;
}

}
