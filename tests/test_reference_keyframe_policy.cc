#include <gtest/gtest.h>

#include "core/heuristic_reference_keyframe_policy.h"

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

TEST(ReferenceKeyframePolicyTest, LateSparseMonoWithStrongCoverageCanRefreshReference) {
    HeuristicReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 22;
    input.detected_keypoints = 852;
    input.candidate_landmarks = 22;
    input.frames_since_reference = 4;
    input.has_depth = false;

    const auto decision = policy.evaluate(input);
    EXPECT_TRUE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, LateSparseMonoNeedsAtLeastThreeFramesToRefresh) {
    HeuristicReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 22;
    input.detected_keypoints = 852;
    input.candidate_landmarks = 22;
    input.frames_since_reference = 2;
    input.has_depth = false;

    const auto decision = policy.evaluate(input);
    EXPECT_FALSE(decision.promoteNewReference());
}

TEST(ReferenceKeyframePolicyTest, LateSparseMonoNeedsEnoughCandidateLandmarksToRefresh) {
    HeuristicReferenceKeyframePolicy policy;
    ReferenceKeyframePolicyInput input;
    input.tracked_features = 22;
    input.detected_keypoints = 852;
    input.candidate_landmarks = 14;
    input.frames_since_reference = 4;
    input.has_depth = false;

    const auto decision = policy.evaluate(input);
    EXPECT_FALSE(decision.promoteNewReference());
}

