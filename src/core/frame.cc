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

void Frame::extractORB(const cv::Ptr<cv::Feature2D>& detector) {
    detector->detectAndCompute(image_, cv::noArray(), keypoints_, descriptors_);
    landmarks_.resize(keypoints_.size(), nullptr);
}

}
