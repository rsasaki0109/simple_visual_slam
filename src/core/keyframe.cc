#include "core/keyframe.h"
#include "core/landmark.h"
#include <algorithm>

namespace svslam {

Keyframe::Keyframe(Frame::Ptr frame)
    : id_(frame->id_), timestamp_(frame->timestamp_), camera_(frame->camera_),
      T_cw_(frame->getPose()), keypoints_(frame->keypoints_), descriptors_(frame->descriptors_.clone()),
      landmarks_(frame->landmarks_)
{
    // Deep copy or shared resources management logic here
}

void Keyframe::updateConnections() {
    std::map<Keyframe::Ptr, int> keyframe_weights;
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    for (auto& lm : landmarks_) {
        if (!lm) continue;
        if (lm->isBad()) continue; // Assuming isBad() exists or check if valid
        
        std::map<std::weak_ptr<Keyframe>, size_t, std::owner_less<std::weak_ptr<Keyframe>>> observations;
        {
             std::unique_lock<std::mutex> lm_lock(lm->mutex_);
             observations = lm->observations_;
        }
        
        for (auto& obs : observations) {
            auto kf = obs.first.lock();
            if (kf && kf->id_ != id_) {
                keyframe_weights[kf]++;
            }
        }
    }
    
    connected_keyframes_.clear();
    // Filter connections (e.g., weight > 15)
    for (auto& kv : keyframe_weights) {
        if (kv.second > 15) {
            connected_keyframes_[kv.first] = kv.second;
            // Also update the other keyframe
            kv.first->addConnection(shared_from_this(), kv.second);
        }
    }
}

void Keyframe::addConnection(Keyframe::Ptr kf, int weight) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!connected_keyframes_.count(kf)) {
        connected_keyframes_[kf] = weight;
    } else if (connected_keyframes_[kf] != weight) {
         connected_keyframes_[kf] = weight;
    }
}

std::vector<Keyframe::Ptr> Keyframe::getBestCovisibilityKeyframes(int N) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::pair<int, Keyframe::Ptr>> pairs;
    for (auto& kv : connected_keyframes_) {
        pairs.push_back({kv.second, kv.first});
    }
    
    std::sort(pairs.rbegin(), pairs.rend()); // Sort by weight descending
    
    std::vector<Keyframe::Ptr> res;
    for (size_t i = 0; i < pairs.size() && i < (size_t)N; ++i) {
        res.push_back(pairs[i].second);
    }
    return res;
}

}
