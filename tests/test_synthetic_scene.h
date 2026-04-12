#pragma once

#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "core/camera.h"
#include "core/frame.h"

namespace svslam::test_support {

constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;

inline Camera::Ptr makeTestCamera() {
    return std::make_shared<Camera>(517.3, 516.5, 318.6, 255.3);
}

inline std::vector<Vec3> makeSyntheticWorldPoints() {
    std::vector<Vec3> points;
    points.reserve(120);
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 12; ++col) {
            const double x = -1.1 + 0.2 * static_cast<double>(col);
            const double y = -0.8 + 0.16 * static_cast<double>(row);
            const double z = 4.0 + 0.15 * static_cast<double>((row + col) % 4);
            points.emplace_back(x, y, z);
        }
    }
    return points;
}

inline cv::Mat makeDescriptors(int rows) {
    cv::Mat descriptors = cv::Mat::zeros(rows, 32, CV_8U);
    for (int row = 0; row < rows; ++row) {
        for (int offset = 0; offset < 4; ++offset) {
            const int bit_index = (row * 31 + offset * 67) % 256;
            descriptors.at<unsigned char>(row, bit_index / 8) |=
                static_cast<unsigned char>(1u << (bit_index % 8));
        }
    }
    return descriptors;
}

inline Frame::Ptr makeProjectedFrame(unsigned long id,
                                     double timestamp,
                                     const Camera::Ptr& camera,
                                     const SE3& pose,
                                     const std::vector<Vec3>& world_points,
                                     bool with_depth = false) {
    auto frame = std::make_shared<Frame>(
        id, timestamp, camera, cv::Mat::zeros(kImageHeight, kImageWidth, CV_8UC1));
    frame->setPose(pose);

    if (with_depth) {
        frame->depth_image_ = cv::Mat::zeros(kImageHeight, kImageWidth, CV_32FC1);
    }

    frame->keypoints_.reserve(world_points.size());
    for (std::size_t index = 0; index < world_points.size(); ++index) {
        const Vec3 point_camera = pose * world_points[index];
        const Vec2 pixel = camera->project(point_camera);
        frame->keypoints_.emplace_back(
            cv::Point2f(static_cast<float>(pixel.x()), static_cast<float>(pixel.y())), 20.0f);
        frame->keypoints_.back().octave = static_cast<int>(index % 4);
        if (with_depth) {
            const int x = static_cast<int>(std::round(pixel.x()));
            const int y = static_cast<int>(std::round(pixel.y()));
            frame->depth_image_.at<float>(y, x) = static_cast<float>(point_camera.z());
        }
    }

    frame->descriptors_ = makeDescriptors(static_cast<int>(frame->keypoints_.size()));
    frame->landmarks_.assign(frame->keypoints_.size(), nullptr);
    return frame;
}

}  // namespace svslam::test_support
