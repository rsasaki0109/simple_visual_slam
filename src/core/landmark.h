#pragma once

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
    
    bool isBad() const { return is_bad_; }
    void setBad() { is_bad_ = true; }

    std::mutex mutex_;
    
private:
    bool is_bad_ = false;
};

}
