#include <gtest/gtest.h>

#include "core/camera.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include "core/map.h"
#include "loop_closing/loop_closing.h"

using namespace svslam;

namespace {

Camera::Ptr makeCamera() {
    return std::make_shared<Camera>(517.3, 516.5, 318.6, 255.3);
}

Keyframe::Ptr makeKeyframe(unsigned long id, int slots = 8) {
    auto frame = std::make_shared<Frame>(
        id, 0.0, makeCamera(), cv::Mat::zeros(240, 320, CV_8UC1));
    frame->keypoints_.resize(slots);
    frame->landmarks_.resize(slots);
    return std::make_shared<Keyframe>(frame);
}

}  // namespace

TEST(LoopClosingTest, MergeLandmarksTransfersObservationsAndRemovesSource) {
    auto map = std::make_shared<Map>();
    auto keyframe_a = makeKeyframe(1);
    auto keyframe_b = makeKeyframe(2);

    auto target = std::make_shared<Landmark>(10, Vec3(0.0, 0.0, 4.0));
    auto source = std::make_shared<Landmark>(11, Vec3(0.1, 0.0, 4.0));
    map->addLandmark(target);
    map->addLandmark(source);

    keyframe_a->landmarks_[0] = source;
    keyframe_b->landmarks_[1] = source;
    source->addObservation(keyframe_a, 0);
    source->addObservation(keyframe_b, 1);

    loop_closing_internal::mergeLandmarks(map, target, source);

    EXPECT_TRUE(source->isBad());
    EXPECT_EQ(map->getAllLandmarks().count(source->id_), 0u);
    ASSERT_EQ(keyframe_a->landmarks_[0], target);
    ASSERT_EQ(keyframe_b->landmarks_[1], target);
    EXPECT_EQ(target->observations_.size(), 2u);
}

TEST(LoopClosingTest, FinalSim3ScaleValidationDependsOnMetricDepth) {
    EXPECT_TRUE(loop_closing_internal::isFinalSim3ScaleAcceptable(1.20, false));
    EXPECT_TRUE(loop_closing_internal::isFinalSim3ScaleAcceptable(1.03, true));
    EXPECT_FALSE(loop_closing_internal::isFinalSim3ScaleAcceptable(1.08, true));
}

TEST(LoopClosingTest, LoopConstraintWeightingMatchesConfidencePolicy) {
    const auto metric = loop_closing_internal::computeLoopConstraintWeighting(40, 0.60, true);
    EXPECT_DOUBLE_EQ(metric.confidence, 1.0);
    EXPECT_DOUBLE_EQ(metric.translation_weight, 7.0);
    EXPECT_DOUBLE_EQ(metric.rotation_weight, 7.0);
    EXPECT_DOUBLE_EQ(metric.scale_weight, 1000.0);

    const auto mono = loop_closing_internal::computeLoopConstraintWeighting(12, 0.20, false);
    EXPECT_DOUBLE_EQ(mono.translation_weight, 10.0);
    EXPECT_DOUBLE_EQ(mono.rotation_weight, 10.0);
    EXPECT_DOUBLE_EQ(mono.scale_weight, 15.0);
}

TEST(LoopClosingTest, StaleEdgeDecayClampsToSafetyFloor) {
    EXPECT_DOUBLE_EQ(loop_closing_internal::computeStaleLoopEdgeDecay(0.10, 0.01), 1.0);
    EXPECT_DOUBLE_EQ(loop_closing_internal::computeStaleLoopEdgeDecay(0.50, 0.01), 0.35);
    EXPECT_DOUBLE_EQ(loop_closing_internal::computeStaleLoopEdgeDecay(0.10, 0.04), 0.5);
}
