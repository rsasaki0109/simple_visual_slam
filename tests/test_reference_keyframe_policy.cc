#include <gtest/gtest.h>

#include "core/heuristic_reference_keyframe_policy.h"
#include "experiments/reference_keyframe/pipeline_reference_keyframe_policy.h"
#include "experiments/reference_keyframe/score_reference_keyframe_policy.h"

using namespace svslam;

TEST(ReferenceKeyframePolicyTest, SparseMonoStaysOnPreviousReference) {
    HeuristicReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 24;
    input.detected_keypoints = 120;
    input.candidate_landmarks = 18;
    input.has_depth = false;

    const auto decision = policy.evaluate(input);
    EXPECT_FALSE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, DepthCandidateCanPromoteEvenWhenSparse) {
    HeuristicReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 24;
    input.detected_keypoints = 120;
    input.candidate_landmarks = 40;
    input.has_depth = true;

    const auto decision = policy.evaluate(input);
    EXPECT_TRUE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, DenseMonoCandidatePromotes) {
    HeuristicReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 52;
    input.detected_keypoints = 260;
    input.candidate_landmarks = 60;
    input.has_depth = false;

    const auto decision = policy.evaluate(input);
    EXPECT_TRUE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, ScorePolicyPromotesDepthAccelSparseRescue) {
    ScoreReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 23;
    input.detected_keypoints = 126;
    input.candidate_landmarks = 30;
    input.frames_since_reference = 4;
    input.lost_frames = 1;
    input.has_depth = true;
    input.has_accel = true;

    const auto decision = policy.evaluate(input);
    EXPECT_TRUE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, ScorePolicyKeepsDepthAccelThinSupport) {
    ScoreReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 24;
    input.detected_keypoints = 130;
    input.candidate_landmarks = 16;
    input.frames_since_reference = 6;
    input.lost_frames = 1;
    input.has_depth = true;
    input.has_accel = true;

    const auto decision = policy.evaluate(input);
    EXPECT_FALSE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, PipelinePolicyPromotesDepthAccelSparseRescue) {
    PipelineReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 23;
    input.detected_keypoints = 126;
    input.candidate_landmarks = 30;
    input.frames_since_reference = 4;
    input.lost_frames = 1;
    input.has_depth = true;
    input.has_accel = true;

    const auto decision = policy.evaluate(input);
    EXPECT_TRUE(decision.promoteNewReference());
}
