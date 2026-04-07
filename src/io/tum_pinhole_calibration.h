#pragma once

#include <string>
#include <vector>

namespace svslam {

// Pinhole intrinsics (+ optional OpenCV radial distortion) for TUM-style RGB-D sequences.
// Used to override built-in freiburg1 defaults when running on custom cameras.
struct TumPinholeCalibration {
    double fx = 517.3;
    double fy = 516.5;
    double cx = 318.6;
    double cy = 255.3;
    int image_width = 640;
    int image_height = 480;
    // If empty, images are used as-is (no undistortion / remapping).
    std::vector<double> distortion;

    static TumPinholeCalibration fr1_default();

    // Minimal JSON reader (no external JSON library). Expected keys are optional except fx,fy,cx,cy.
    // Optional: "width", "height", "distortion": [ k1, k2, p1, p2, k3, ... ]
    static bool load_json_file(const std::string& path, TumPinholeCalibration& out, std::string& error);
};

}  // namespace svslam
