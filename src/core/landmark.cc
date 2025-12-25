#include "core/landmark.h"

namespace svslam {

Landmark::Landmark(unsigned long id, const Vec3& pos)
    : id_(id), pos_w_(pos) {}

void Landmark::setPos(const Vec3& pos) {
    std::unique_lock<std::mutex> lock(mutex_);
    pos_w_ = pos;
}

Vec3 Landmark::getPos() const {
    // std::unique_lock<std::mutex> lock(mutex_); // Can be locked if needed
    return pos_w_;
}

void Landmark::addObservation(std::shared_ptr<Keyframe> kf, size_t idx_in_kf) {
    std::unique_lock<std::mutex> lock(mutex_);
    observations_[kf] = idx_in_kf;
}

void Landmark::removeObservation(std::shared_ptr<Keyframe> kf) {
    std::unique_lock<std::mutex> lock(mutex_);
    observations_.erase(kf);
}

}
