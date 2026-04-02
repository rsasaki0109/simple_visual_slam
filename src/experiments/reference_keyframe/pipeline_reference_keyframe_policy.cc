#include "experiments/reference_keyframe/pipeline_reference_keyframe_policy.h"

#include <algorithm>

namespace svslam {

namespace {

constexpr int kMonoTrackedFloor = 32;
constexpr int kMonoKeypointFloor = 140;
constexpr int kDepthLandmarkFloor = 35;
constexpr int kAccelDepthLandmarkFloor = 28;
constexpr int kMapSupportFloor = 20;
constexpr int kFreshAnchorFrames = 10;
constexpr int kDenseTrackedFloor = 50;
constexpr int kDenseKeypointFloor = 220;

double confidence(bool strong_signal) {
    return strong_signal ? 0.86 : 0.68;
}

}  // namespace

std::string PipelineReferenceKeyframePolicy::name() const {
    return "pipeline";
}

std::string PipelineReferenceKeyframePolicy::philosophy() const {
    return "staged-gates";
}

bool PipelineReferenceKeyframePolicy::failsSparseMonoGate(const ReferenceKeyframePolicyInput& input) const {
    return !input.has_depth &&
           (input.tracked_features < kMonoTrackedFloor ||
            input.detected_keypoints < kMonoKeypointFloor ||
            input.lost_frames >= 3);
}

bool PipelineReferenceKeyframePolicy::passesDepthOverride(const ReferenceKeyframePolicyInput& input) const {
    const int required_landmarks = input.has_accel ? kAccelDepthLandmarkFloor : kDepthLandmarkFloor;
    return input.has_depth && input.candidate_landmarks >= required_landmarks;
}

bool PipelineReferenceKeyframePolicy::failsMapSupportGate(const ReferenceKeyframePolicyInput& input) const {
    return input.candidate_landmarks < kMapSupportFloor;
}

bool PipelineReferenceKeyframePolicy::needsFreshAnchor(const ReferenceKeyframePolicyInput& input) const {
    return input.frames_since_reference >= kFreshAnchorFrames;
}

bool PipelineReferenceKeyframePolicy::passesDenseTrackingGate(const ReferenceKeyframePolicyInput& input) const {
    return input.tracked_features >= kDenseTrackedFloor &&
           input.detected_keypoints >= kDenseKeypointFloor;
}

ReferenceKeyframeDecision PipelineReferenceKeyframePolicy::evaluate(
    const ReferenceKeyframePolicyInput& input) const {
    if (passesDepthOverride(input)) {
        return {
            ReferenceKeyframeAction::PromoteNewReference,
            confidence(true),
            input.has_accel ? "depth+accel override allows immediate promotion"
                            : "depth override allows immediate promotion"
        };
    }

    if (failsSparseMonoGate(input)) {
        return {
            ReferenceKeyframeAction::KeepCurrentReference,
            confidence(true),
            "mono safety gate vetoed promotion"
        };
    }

    if (failsMapSupportGate(input)) {
        return {
            ReferenceKeyframeAction::KeepCurrentReference,
            confidence(false),
            "candidate map support is too thin"
        };
    }

    if (needsFreshAnchor(input)) {
        return {
            ReferenceKeyframeAction::PromoteNewReference,
            confidence(false),
            "reference has aged out and needs a fresh anchor"
        };
    }

    if (passesDenseTrackingGate(input)) {
        return {
            ReferenceKeyframeAction::PromoteNewReference,
            confidence(true),
            "dense tracking passed the final promotion gate"
        };
    }

    return {
        input.has_depth ? ReferenceKeyframeAction::PromoteNewReference
                        : ReferenceKeyframeAction::KeepCurrentReference,
        confidence(false),
        input.has_depth ? (input.has_accel
                               ? "depth+accel fallback promotes the candidate"
                               : "depth fallback promotes the candidate")
                        : "mono fallback keeps the current reference"
    };
}

}  // namespace svslam
