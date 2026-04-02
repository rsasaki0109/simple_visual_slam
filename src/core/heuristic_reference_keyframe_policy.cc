#include "core/heuristic_reference_keyframe_policy.h"

#include <algorithm>
#include <cmath>

namespace svslam {

namespace {

constexpr int kMinTrackedFeatures = 35;
constexpr int kMinDetectedKeypoints = 150;

double clampConfidence(double value) {
    return std::clamp(value, 0.05, 0.99);
}

}  // namespace

std::string HeuristicReferenceKeyframePolicy::name() const {
    return "heuristic";
}

std::string HeuristicReferenceKeyframePolicy::philosophy() const {
    return "imperative-thresholds";
}

ReferenceKeyframeDecision HeuristicReferenceKeyframePolicy::evaluate(
    const ReferenceKeyframePolicyInput& input) const {
    const bool sparse_mono_frame =
        !input.has_depth &&
        (input.tracked_features < kMinTrackedFeatures ||
         input.detected_keypoints < kMinDetectedKeypoints);

    if (sparse_mono_frame) {
        const double tracked_margin = static_cast<double>(kMinTrackedFeatures - input.tracked_features) /
                                      static_cast<double>(kMinTrackedFeatures);
        const double keypoint_margin = static_cast<double>(kMinDetectedKeypoints - input.detected_keypoints) /
                                       static_cast<double>(kMinDetectedKeypoints);
        return {
            ReferenceKeyframeAction::KeepCurrentReference,
            clampConfidence(0.55 + std::max(tracked_margin, keypoint_margin)),
            "mono frame is too sparse for immediate promotion"
        };
    }

    const double tracked_margin = static_cast<double>(input.tracked_features - kMinTrackedFeatures) /
                                  static_cast<double>(std::max(1, kMinTrackedFeatures));
    const double keypoint_margin = static_cast<double>(input.detected_keypoints - kMinDetectedKeypoints) /
                                   static_cast<double>(std::max(1, kMinDetectedKeypoints));
    return {
        ReferenceKeyframeAction::PromoteNewReference,
        clampConfidence(0.60 + 0.5 * std::max(0.0, std::max(tracked_margin, keypoint_margin))),
        input.has_depth ? "depth-backed keyframe can become the new reference"
                        : "mono frame cleared the sparse-frame veto"
    };
}

}  // namespace svslam
