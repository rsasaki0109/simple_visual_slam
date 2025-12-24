#pragma once

#include "core/common.h"
#include <unordered_map>

namespace svslam {

class Map {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Map>;

    void addKeyframe(std::shared_ptr<Keyframe> kf);
    void addLandmark(std::shared_ptr<Landmark> lm);
    
    void removeKeyframe(std::shared_ptr<Keyframe> kf);
    void removeLandmark(std::shared_ptr<Landmark> lm);

    const std::map<unsigned long, std::shared_ptr<Keyframe>>& getAllKeyframes() const;
    const std::map<unsigned long, std::shared_ptr<Landmark>>& getAllLandmarks() const;

    void clear();

    std::mutex mutex_;
    
private:
    std::map<unsigned long, std::shared_ptr<Keyframe>> keyframes_;
    std::map<unsigned long, std::shared_ptr<Landmark>> landmarks_;
};

}
