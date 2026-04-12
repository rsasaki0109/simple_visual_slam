#include <gtest/gtest.h>

#include "tracking/tracking.h"

using namespace svslam;

TEST(TrackingPoseRecomputeTest, AcceptsOnlyStrictReprojectionImprovement) {
    EXPECT_TRUE(Tracking::shouldAcceptRecomputedPose(1.90, 1.85));
    EXPECT_FALSE(Tracking::shouldAcceptRecomputedPose(1.90, 1.90));
    EXPECT_FALSE(Tracking::shouldAcceptRecomputedPose(1.90, 2.15));
}

TEST(TrackingPoseRecomputeTest, RejectsFormerAbsoluteFallbackCase) {
    EXPECT_FALSE(Tracking::shouldAcceptRecomputedPose(1.90, 19.0));
}

TEST(TrackingPoseRecomputeTest, AcceptsLocalMapUpdatesOutsideRecoveryWindow) {
    EXPECT_TRUE(Tracking::shouldAcceptLocalMapPoseUpdate(
        40, 240, true, 0.30, 0.25, 0));
}

TEST(TrackingPoseRecomputeTest, RejectsThinSupportJumpDuringRecovery) {
    EXPECT_FALSE(Tracking::shouldAcceptLocalMapPoseUpdate(
        89, 537, false, 0.33, 0.26, 2));
    EXPECT_FALSE(Tracking::shouldAcceptLocalMapPoseUpdate(
        37, 257, true, 0.33, 0.25, 2));
}

TEST(TrackingPoseRecomputeTest, RejectsLargeJumpEvenWithModerateSupportDuringRecovery) {
    EXPECT_FALSE(Tracking::shouldAcceptLocalMapPoseUpdate(
        191, 246, false, 0.24, 0.27, 1));
}

TEST(TrackingPoseRecomputeTest, AcceptsStrongStableUpdateDuringRecovery) {
    EXPECT_TRUE(Tracking::shouldAcceptLocalMapPoseUpdate(
        966, 966, false, 0.06, 0.04, 2));
    EXPECT_TRUE(Tracking::shouldAcceptLocalMapPoseUpdate(
        344, 217, false, 0.03, 0.03, 1));
}

TEST(TrackingPoseRecomputeTest, RelocalizationCandidatePolicyStaysLocalDuringLoopRecovery) {
    EXPECT_TRUE(Tracking::shouldConsiderRelocalizationCandidate(
        3.5, true, true, 2));
    EXPECT_TRUE(Tracking::shouldConsiderRelocalizationCandidate(
        2.0, false, true, 2));
    EXPECT_FALSE(Tracking::shouldConsiderRelocalizationCandidate(
        3.0, false, true, 2));
}

TEST(TrackingPoseRecomputeTest, RelocalizationCandidatePolicyRelaxesAfterLoopPendingClears) {
    EXPECT_TRUE(Tracking::shouldConsiderRelocalizationCandidate(
        3.5, false, false, 1));
    EXPECT_FALSE(Tracking::shouldConsiderRelocalizationCandidate(
        4.5, false, false, 1));
    EXPECT_TRUE(Tracking::shouldConsiderRelocalizationCandidate(
        10.0, false, false, 0));
}
