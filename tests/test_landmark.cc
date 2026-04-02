#include <gtest/gtest.h>
#include <thread>
#include "core/landmark.h"
#include "core/keyframe.h"
#include "core/frame.h"
#include "core/camera.h"

using namespace svslam;

// Helper to create a dummy keyframe
static Keyframe::Ptr makeDummyKF(unsigned long id) {
    auto cam = std::make_shared<Camera>(500, 500, 320, 240);
    auto frame = std::make_shared<Frame>(id, 0.0, cam, cv::Mat::zeros(480, 640, CV_8UC1));
    frame->keypoints_.resize(10);
    frame->landmarks_.resize(10);
    return std::make_shared<Keyframe>(frame);
}

TEST(LandmarkTest, AddRemoveObservation) {
    auto lm = std::make_shared<Landmark>(0, Vec3(1.0, 2.0, 3.0));
    auto kf1 = makeDummyKF(0);
    auto kf2 = makeDummyKF(1);

    lm->addObservation(kf1, 0);
    lm->addObservation(kf2, 3);
    EXPECT_EQ(lm->observations_.size(), 2u);

    lm->removeObservation(kf1);
    EXPECT_EQ(lm->observations_.size(), 1u);

    // Removing again should be safe (no-op)
    lm->removeObservation(kf1);
    EXPECT_EQ(lm->observations_.size(), 1u);
}

TEST(LandmarkTest, SetGetPos) {
    auto lm = std::make_shared<Landmark>(0, Vec3(0, 0, 0));

    Vec3 new_pos(1.5, -2.3, 4.7);
    lm->setPos(new_pos);

    Vec3 got = lm->getPos();
    EXPECT_DOUBLE_EQ(got.x(), 1.5);
    EXPECT_DOUBLE_EQ(got.y(), -2.3);
    EXPECT_DOUBLE_EQ(got.z(), 4.7);
}

TEST(LandmarkTest, SetGetPosThreadSafety) {
    auto lm = std::make_shared<Landmark>(0, Vec3(0, 0, 0));

    constexpr int kIterations = 10000;

    std::thread writer([&]() {
        for (int i = 0; i < kIterations; ++i) {
            double v = static_cast<double>(i);
            lm->setPos(Vec3(v, v, v));
        }
    });

    std::thread reader([&]() {
        for (int i = 0; i < kIterations; ++i) {
            Vec3 pos = lm->getPos();
            // Just verify it doesn't crash and values are finite
            EXPECT_TRUE(std::isfinite(pos.x()));
            EXPECT_TRUE(std::isfinite(pos.y()));
            EXPECT_TRUE(std::isfinite(pos.z()));
        }
    });

    writer.join();
    reader.join();
}

TEST(LandmarkTest, IsBadSetBad) {
    auto lm = std::make_shared<Landmark>(0, Vec3(1, 2, 3));

    EXPECT_FALSE(lm->isBad());
    lm->setBad();
    EXPECT_TRUE(lm->isBad());
}
