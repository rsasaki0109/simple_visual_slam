#pragma once

#include "core/reference_keyframe_policy.h"

namespace svslam {

class HeuristicReferenceKeyframePolicy final : public ReferenceKeyframePolicy {
public:
    std::string name() const override;
    std::string philosophy() const override;
    ReferenceKeyframeDecision evaluate(const ReferenceKeyframePolicyInput& input) const override;
};

}  // namespace svslam
