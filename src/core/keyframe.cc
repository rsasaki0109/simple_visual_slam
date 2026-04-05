#include "core/keyframe.h"
#include "core/landmark.h"
#include <algorithm>

namespace svslam {

Keyframe::Keyframe(Frame::Ptr frame)
    : id_(frame->id_), timestamp_(frame->timestamp_), camera_(frame->camera_),
      T_cw_(frame->getPose()),
      depth_image_(frame->depth_image_.empty() ? cv::Mat() : frame->depth_image_.clone()),
      depth_is_metric_(frame->depth_is_metric_),
      keypoints_(frame->keypoints_), descriptors_(frame->descriptors_.clone()),
      landmarks_(frame->landmarks_)
{
}

float Keyframe::getDepth(float u, float v) const {
    if (depth_image_.empty()) return -1.0f;

    int x = static_cast<int>(std::round(u));
    int y = static_cast<int>(std::round(v));
    if (x < 0 || x >= depth_image_.cols || y < 0 || y >= depth_image_.rows)
        return -1.0f;

    if (depth_image_.type() == CV_16UC1) {
        uint16_t raw = depth_image_.at<uint16_t>(y, x);
        if (raw == 0) return -1.0f;
        return static_cast<float>(raw) / 5000.0f;
    } else if (depth_image_.type() == CV_32FC1) {
        float d = depth_image_.at<float>(y, x);
        if (d <= 0.0f || !std::isfinite(d)) return -1.0f;
        return d;
    }
    return -1.0f;
}

void Keyframe::updateConnections() {
    std::map<Keyframe::Ptr, int> keyframe_weights;

    std::vector<Landmark::Ptr> landmarks_copy;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        landmarks_copy = landmarks_;
    }

    for (auto& lm : landmarks_copy) {
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

    std::map<Keyframe::Ptr, int> filtered_connections;
    // Filter connections (e.g., weight > 15)
    for (auto& kv : keyframe_weights) {
        if (kv.second > 15) {
            filtered_connections[kv.first] = kv.second;
        }
    }

    {
        std::unique_lock<std::mutex> lock(mutex_);
        connected_keyframes_ = filtered_connections;
    }

    for (auto& kv : filtered_connections) {
        kv.first->addConnection(shared_from_this(), kv.second);
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
    pairs.reserve(connected_keyframes_.size());
    for (auto& kv : connected_keyframes_) {
        pairs.push_back({kv.second, kv.first});
    }

    // Weight desc, then keyframe id asc — avoid std::shared_ptr address ordering on ties
    // (pointer order varies run-to-run and breaks reproducible local mapping / BA inputs).
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int, Keyframe::Ptr>& a, const std::pair<int, Keyframe::Ptr>& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  const unsigned long id_a = a.second ? a.second->id_ : 0UL;
                  const unsigned long id_b = b.second ? b.second->id_ : 0UL;
                  return id_a < id_b;
              });

    std::vector<Keyframe::Ptr> res;
    for (size_t i = 0; i < pairs.size() && i < (size_t)N; ++i) {
        res.push_back(pairs[i].second);
    }
    return res;
}

}
