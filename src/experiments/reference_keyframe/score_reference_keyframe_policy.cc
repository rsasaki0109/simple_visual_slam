#include "experiments/reference_keyframe/score_reference_keyframe_policy.h"

#include <algorithm>
#include <cmath>

namespace svslam {

namespace {

constexpr double kCoverageWeight = 0.35;
constexpr double kMapSupportWeight = 0.25;
constexpr double kFreshnessWeight = 0.15;
constexpr double kDepthBonus = 0.20;
constexpr double kAccelSupportBonus = 0.08;
constexpr double kLostPenaltyScale = 0.12;
constexpr double kAccelLostPenaltyScale = 0.08;
constexpr double kPromoteThreshold = 0.55;
constexpr int kAccelSupportLandmarkFloor = 24;

double normalizeRatio(int value, int target) {
    return std::clamp(static_cast<double>(value) / static_cast<double>(std::max(1, target)), 0.0, 1.25);
}

double clampConfidence(double value) {
    return std::clamp(value, 0.05, 0.99);
}

}  // namespace

std::string ScoreReferenceKeyframePolicy::name() const {
    return "score";
}

std::string ScoreReferenceKeyframePolicy::philosophy() const {
    return "weighted-score";
}

double ScoreReferenceKeyframePolicy::coverageScore(const ReferenceKeyframePolicyInput& input) const {
    const double tracked = normalizeRatio(input.tracked_features, 55);
    const double keypoints = normalizeRatio(input.detected_keypoints, 220);
    return 0.55 * tracked + 0.45 * keypoints;
}

double ScoreReferenceKeyframePolicy::mapSupportScore(const ReferenceKeyframePolicyInput& input) const {
    return normalizeRatio(input.candidate_landmarks, 55);
}

double ScoreReferenceKeyframePolicy::freshnessScore(const ReferenceKeyframePolicyInput& input) const {
    return std::clamp(static_cast<double>(input.frames_since_reference) / 12.0, 0.0, 1.0);
}

double ScoreReferenceKeyframePolicy::stabilityPenalty(const ReferenceKeyframePolicyInput& input) const {
    const double scale = input.has_accel ? kAccelLostPenaltyScale : kLostPenaltyScale;
    return std::clamp(static_cast<double>(input.lost_frames) * scale, 0.0, 0.5);
}

ReferenceKeyframeDecision ScoreReferenceKeyframePolicy::evaluate(
    const ReferenceKeyframePolicyInput& input) const {
    double score = 0.0;
    score += kCoverageWeight * coverageScore(input);
    score += kMapSupportWeight * mapSupportScore(input);
    score += kFreshnessWeight * freshnessScore(input);

    if (input.has_depth) {
        score += kDepthBonus;
    }
    if (input.has_accel && input.candidate_landmarks >= kAccelSupportLandmarkFloor) {
        score += kAccelSupportBonus;
    }
    if (!input.has_depth && input.detected_keypoints < 140) {
        score -= 0.10;
    }

    score -= stabilityPenalty(input);

    const bool promote = score >= kPromoteThreshold;
    const double confidence = clampConfidence(0.55 + std::fabs(score - kPromoteThreshold));
    return {
        promote ? ReferenceKeyframeAction::PromoteNewReference
                : ReferenceKeyframeAction::KeepCurrentReference,
        confidence,
        promote ? (input.has_accel
                       ? "weighted score cleared the promotion threshold with accel support"
                       : "weighted score cleared the promotion threshold")
                : "weighted score stayed below the promotion threshold"
    };
}

}  // namespace svslam
