#pragma once

#include <string>
#include <vector>

namespace svslam {

struct EurocPinholeCalibration {
    struct Camera {
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        int image_width = 752;
        int image_height = 480;
        std::vector<double> distortion;
    };

    Camera cam0;
    Camera cam1;
    bool has_cam1 = false;

    // Minimal JSON reader (no external JSON library).
    // Accepted forms:
    //   { "fx": ..., "fy": ..., "cx": ..., "cy": ..., "distortion": [...] }
    //   { "cam0": { ... }, "cam1": { ... } }
    // Optional keys per camera: "width", "height", "distortion".
    static bool load_json_file(const std::string& path, EurocPinholeCalibration& out, std::string& error);
};

}  // namespace svslam
