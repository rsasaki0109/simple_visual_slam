#include <gtest/gtest.h>
#include "core/camera.h"

using namespace svslam;

// TUM freiburg1 parameters
static constexpr double kFx = 517.3;
static constexpr double kFy = 516.5;
static constexpr double kCx = 318.6;
static constexpr double kCy = 255.3;

TEST(CameraTest, ProjectUnprojectRoundtrip) {
    auto cam = std::make_shared<Camera>(kFx, kFy, kCx, kCy);

    // Test several known pixel coordinates
    std::vector<Vec2> pixels = {
        {320.0, 240.0},
        {0.0, 0.0},
        {640.0, 480.0},
        {kCx, kCy},  // principal point
        {100.5, 300.7},
    };

    for (const auto& px : pixels) {
        // Unproject to normalized coords (z=1), then create a 3D point at some depth
        Vec3 ray = cam->unproject(px);
        double depth = 3.5;
        Vec3 point_3d = ray * depth;

        // Project back to pixel
        Vec2 reprojected = cam->project(point_3d);

        EXPECT_NEAR(reprojected.x(), px.x(), 1e-10)
            << "Failed for pixel (" << px.x() << ", " << px.y() << ")";
        EXPECT_NEAR(reprojected.y(), px.y(), 1e-10)
            << "Failed for pixel (" << px.x() << ", " << px.y() << ")";
    }
}

TEST(CameraTest, ProjectUnprojectPrincipalPoint) {
    auto cam = std::make_shared<Camera>(kFx, kFy, kCx, kCy);

    // The principal point should unproject to (0,0,1) in normalized coords
    Vec3 ray = cam->unproject(Vec2(kCx, kCy));
    EXPECT_NEAR(ray.x(), 0.0, 1e-10);
    EXPECT_NEAR(ray.y(), 0.0, 1e-10);
    EXPECT_NEAR(ray.z(), 1.0, 1e-10);
}

TEST(CameraTest, KMatrixValues) {
    auto cam = std::make_shared<Camera>(kFx, kFy, kCx, kCy);
    cv::Mat K = cam->K();

    EXPECT_DOUBLE_EQ(K.at<double>(0, 0), kFx);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 1), kFy);
    EXPECT_DOUBLE_EQ(K.at<double>(0, 2), kCx);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 2), kCy);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 2), 1.0);
}
