#include "core/frame.h"

namespace svslam {

Frame::Frame(unsigned long id, double timestamp, Camera::Ptr camera, const cv::Mat& image)
    : id_(id), timestamp_(timestamp), camera_(camera), image_(image) {}

void Frame::setPose(const SE3& pose) {
    std::unique_lock<std::mutex> lock(mutex_);
    T_cw_ = pose;
}

SE3 Frame::getPose() const {
    // std::unique_lock<std::mutex> lock(mutex_);
    return T_cw_;
}

float Frame::getDepth(float u, float v) const {
    if (depth_image_.empty()) return -1.0f;

    int x = static_cast<int>(std::round(u));
    int y = static_cast<int>(std::round(v));
    if (x < 0 || x >= depth_image_.cols || y < 0 || y >= depth_image_.rows)
        return -1.0f;

    if (depth_image_.type() == CV_16UC1) {
        uint16_t raw = depth_image_.at<uint16_t>(y, x);
        if (raw == 0) return -1.0f;
        return static_cast<float>(raw) / 5000.0f;  // TUM scale factor
    } else if (depth_image_.type() == CV_32FC1) {
        float d = depth_image_.at<float>(y, x);
        if (d <= 0.0f || !std::isfinite(d)) return -1.0f;
        return d;
    }
    return -1.0f;
}

Vec3 Frame::backprojectWithDepth(const cv::KeyPoint& kp, float depth_m) const {
    if (!camera_ || depth_m <= 0.0f) return Vec3(0, 0, 0);
    Vec3 p_norm = camera_->unproject(Vec2(kp.pt.x, kp.pt.y));
    Vec3 p_cam = p_norm * depth_m;
    // Transform to world: P_w = T_wc * P_c
    SE3 T_wc = T_cw_.inverse();
    return T_wc * p_cam;
}

void Frame::extractORB(const cv::Ptr<cv::Feature2D>& detector) {
    detector->detectAndCompute(image_, cv::noArray(), keypoints_, descriptors_);
    landmarks_.resize(keypoints_.size(), nullptr);
}

}
