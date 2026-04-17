#pragma once

#include <atomic>

#include "core/common.h"

namespace svslam {

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Landmark>;

    Landmark(unsigned long id, const Vec3& pos);

    void setPos(const Vec3& pos);
    Vec3 getPos() const;

    void addObservation(std::shared_ptr<Keyframe> kf, size_t idx_in_kf);
    void removeObservation(std::shared_ptr<Keyframe> kf);

    unsigned long id_;
    Vec3 pos_w_; // World position

    // Observations: Keyframe -> index of feature
    // Use weak_ptr to avoid circular dependency (Keyframe -> Landmark -> Keyframe)
    std::map<std::weak_ptr<Keyframe>, size_t, std::owner_less<std::weak_ptr<Keyframe>>> observations_;

    // Representative descriptor (for matching)
    cv::Mat descriptor_;

    // is_bad_ is written by LocalMapping (BA culling) while Tracking reads it
    // from needNewKeyframe and trackLocalMap. Using std::atomic<bool> gives a
    // race-free publish without taking mutex_, which the hot-path readers don't
    // hold. TSan flagged this as the top race in the async pipeline.
    bool isBad() const { return is_bad_.load(std::memory_order_acquire); }
    void setBad() { is_bad_.store(true, std::memory_order_release); }

    mutable std::mutex mutex_;

private:
    std::atomic<bool> is_bad_{false};
};

}
