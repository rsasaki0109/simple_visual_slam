#include <gtest/gtest.h>

#include "core/camera.h"
#include "core/keyframe.h"
#include "core/landmark.h"

using namespace svslam;

namespace {

Camera::Ptr makeCamera() {
    return std::make_shared<Camera>(517.3, 516.5, 318.6, 255.3);
}

Keyframe::Ptr makeKeyframe(unsigned long id, int num_slots = 40) {
    auto frame = std::make_shared<Frame>(
        id, 0.0, makeCamera(), cv::Mat::zeros(240, 320, CV_8UC1));
    frame->keypoints_.resize(num_slots);
    frame->landmarks_.resize(num_slots);
    return std::make_shared<Keyframe>(frame);
}

void attachSharedLandmarks(const Keyframe::Ptr& a,
                           const Keyframe::Ptr& b,
                           int count,
                           unsigned long starting_id,
                           int keypoint_offset = 0) {
    for (int i = 0; i < count; ++i) {
        auto landmark = std::make_shared<Landmark>(
            starting_id + static_cast<unsigned long>(i),
            Vec3(0.1 * i, 0.2, 4.0));
        const int keypoint_index = keypoint_offset + i;
        a->landmarks_[keypoint_index] = landmark;
        b->landmarks_[keypoint_index] = landmark;
        landmark->addObservation(a, keypoint_index);
        landmark->addObservation(b, keypoint_index);
    }
}

}  // namespace

TEST(KeyframeTest, UpdateConnectionsCountsSharedObservations) {
    const auto reference = makeKeyframe(1);
    const auto strong_neighbor = makeKeyframe(2);
    const auto weak_neighbor = makeKeyframe(3);

    attachSharedLandmarks(reference, strong_neighbor, 20, 1000, 0);
    attachSharedLandmarks(reference, weak_neighbor, 16, 2000, 20);

    reference->updateConnections();

    ASSERT_EQ(reference->connected_keyframes_.size(), 2u);
    EXPECT_EQ(reference->connected_keyframes_.at(strong_neighbor), 20);
    EXPECT_EQ(reference->connected_keyframes_.at(weak_neighbor), 16);
    EXPECT_EQ(strong_neighbor->connected_keyframes_.at(reference), 20);
    EXPECT_EQ(weak_neighbor->connected_keyframes_.at(reference), 16);
}

TEST(KeyframeTest, GetBestCovisibilityKeyframesSortsByWeightThenId) {
    const auto base = makeKeyframe(1);
    const auto id_two = makeKeyframe(2);
    const auto id_three = makeKeyframe(3);
    const auto id_four = makeKeyframe(4);

    base->addConnection(id_four, 12);
    base->addConnection(id_three, 25);
    base->addConnection(id_two, 25);

    const auto best = base->getBestCovisibilityKeyframes(3);
    ASSERT_EQ(best.size(), 3u);
    EXPECT_EQ(best[0]->id_, 2u);
    EXPECT_EQ(best[1]->id_, 3u);
    EXPECT_EQ(best[2]->id_, 4u);
}
