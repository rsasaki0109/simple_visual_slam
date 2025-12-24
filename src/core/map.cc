#include "core/map.h"
#include "core/keyframe.h"
#include "core/landmark.h"

namespace svslam {

void Map::addKeyframe(std::shared_ptr<Keyframe> kf) {
    std::unique_lock<std::mutex> lock(mutex_);
    keyframes_[kf->id_] = kf;
}

void Map::addLandmark(std::shared_ptr<Landmark> lm) {
    std::unique_lock<std::mutex> lock(mutex_);
    landmarks_[lm->id_] = lm;
}

void Map::removeKeyframe(std::shared_ptr<Keyframe> kf) {
    std::unique_lock<std::mutex> lock(mutex_);
    keyframes_.erase(kf->id_);
}

void Map::removeLandmark(std::shared_ptr<Landmark> lm) {
    std::unique_lock<std::mutex> lock(mutex_);
    landmarks_.erase(lm->id_);
}

const std::map<unsigned long, std::shared_ptr<Keyframe>>& Map::getAllKeyframes() const {
    return keyframes_;
}

const std::map<unsigned long, std::shared_ptr<Landmark>>& Map::getAllLandmarks() const {
    return landmarks_;
}

void Map::clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    keyframes_.clear();
    landmarks_.clear();
}

}
