#pragma once

#include "core/reference_keyframe_policy.h"

namespace svslam {

class PipelineReferenceKeyframePolicy final : public ReferenceKeyframePolicy {
public:
    std::string name() const override;
    std::string philosophy() const override;
    ReferenceKeyframeDecision evaluate(const ReferenceKeyframePolicyInput& input) const override;

private:
    bool failsSparseMonoGate(const ReferenceKeyframePolicyInput& input) const;
    bool passesDepthOverride(const ReferenceKeyframePolicyInput& input) const;
    bool failsMapSupportGate(const ReferenceKeyframePolicyInput& input) const;
    bool needsFreshAnchor(const ReferenceKeyframePolicyInput& input) const;
    bool passesDenseTrackingGate(const ReferenceKeyframePolicyInput& input) const;
};

}  // namespace svslam
