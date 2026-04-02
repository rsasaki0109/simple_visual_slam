#pragma once

#include <string>

namespace svslam {

enum class ReferenceKeyframeAction {
    PromoteNewReference,
    KeepCurrentReference
};

struct ReferenceKeyframePolicyInput {
    int tracked_features = 0;
    int detected_keypoints = 0;
    int candidate_landmarks = 0;
    int frames_since_reference = 0;
    int lost_frames = 0;
    bool has_depth = false;
    bool has_accel = false;
};

struct ReferenceKeyframeDecision {
    ReferenceKeyframeAction action = ReferenceKeyframeAction::KeepCurrentReference;
    double confidence = 0.0;
    std::string reason;

    bool promoteNewReference() const {
        return action == ReferenceKeyframeAction::PromoteNewReference;
    }
};

inline std::string toString(ReferenceKeyframeAction action) {
    return action == ReferenceKeyframeAction::PromoteNewReference ? "promote" : "keep";
}

class ReferenceKeyframePolicy {
public:
    virtual ~ReferenceKeyframePolicy() = default;

    virtual std::string name() const = 0;
    virtual std::string philosophy() const = 0;
    virtual ReferenceKeyframeDecision evaluate(const ReferenceKeyframePolicyInput& input) const = 0;
};

}  // namespace svslam
