#include <gtest/gtest.h>
#include <thread>
#include "core/map.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include "core/frame.h"
#include "core/camera.h"

using namespace svslam;

static Keyframe::Ptr makeDummyKF(unsigned long id) {
    auto cam = std::make_shared<Camera>(500, 500, 320, 240);
    auto frame = std::make_shared<Frame>(id, 0.0, cam, cv::Mat::zeros(480, 640, CV_8UC1));
    frame->keypoints_.resize(10);
    frame->landmarks_.resize(10);
    return std::make_shared<Keyframe>(frame);
}

TEST(MapTest, AddKeyframe) {
    auto map = std::make_shared<Map>();
    auto kf = makeDummyKF(0);

    map->addKeyframe(kf);
    EXPECT_EQ(map->getAllKeyframes().size(), 1u);
}

TEST(MapTest, AddLandmark) {
    auto map = std::make_shared<Map>();
    auto lm = std::make_shared<Landmark>(0, Vec3(1, 2, 3));

    map->addLandmark(lm);
    EXPECT_EQ(map->getAllLandmarks().size(), 1u);
}

TEST(MapTest, RemoveKeyframe) {
    auto map = std::make_shared<Map>();
    auto kf0 = makeDummyKF(0);
    auto kf1 = makeDummyKF(1);

    map->addKeyframe(kf0);
    map->addKeyframe(kf1);
    EXPECT_EQ(map->getAllKeyframes().size(), 2u);

    map->removeKeyframe(kf0);
    EXPECT_EQ(map->getAllKeyframes().size(), 1u);

    // Check the remaining one is kf1
    EXPECT_TRUE(map->getAllKeyframes().count(1));
}

TEST(MapTest, RemoveLandmark) {
    auto map = std::make_shared<Map>();
    auto lm0 = std::make_shared<Landmark>(0, Vec3(1, 0, 0));
    auto lm1 = std::make_shared<Landmark>(1, Vec3(0, 1, 0));

    map->addLandmark(lm0);
    map->addLandmark(lm1);
    EXPECT_EQ(map->getAllLandmarks().size(), 2u);

    map->removeLandmark(lm0);
    EXPECT_EQ(map->getAllLandmarks().size(), 1u);
    EXPECT_TRUE(map->getAllLandmarks().count(1));
}

TEST(MapTest, GetAllKeyframesCount) {
    auto map = std::make_shared<Map>();

    for (unsigned long i = 0; i < 5; ++i) {
        map->addKeyframe(makeDummyKF(i));
    }

    EXPECT_EQ(map->getAllKeyframes().size(), 5u);
}

TEST(MapTest, ConcurrentAddKeyframe) {
    auto map = std::make_shared<Map>();
    constexpr int kPerThread = 50;

    std::thread t1([&]() {
        for (int i = 0; i < kPerThread; ++i) {
            map->addKeyframe(makeDummyKF(static_cast<unsigned long>(i)));
        }
    });

    std::thread t2([&]() {
        for (int i = kPerThread; i < 2 * kPerThread; ++i) {
            map->addKeyframe(makeDummyKF(static_cast<unsigned long>(i)));
        }
    });

    t1.join();
    t2.join();

    // All keyframes should be present (IDs are unique across threads)
    EXPECT_EQ(map->getAllKeyframes().size(), static_cast<size_t>(2 * kPerThread));
}
