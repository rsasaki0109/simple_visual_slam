#pragma once

#include "core/reference_keyframe_policy.h"

namespace svslam {

class ScoreReferenceKeyframePolicy final : public ReferenceKeyframePolicy {
public:
    std::string name() const override;
    std::string philosophy() const override;
    ReferenceKeyframeDecision evaluate(const ReferenceKeyframePolicyInput& input) const override;

private:
    double coverageScore(const ReferenceKeyframePolicyInput& input) const;
    double mapSupportScore(const ReferenceKeyframePolicyInput& input) const;
    double freshnessScore(const ReferenceKeyframePolicyInput& input) const;
    double stabilityPenalty(const ReferenceKeyframePolicyInput& input) const;
};

}  // namespace svslam
