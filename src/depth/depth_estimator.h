#pragma once

#include <opencv2/core.hpp>

namespace svslam {

class DepthEstimator {
public:
    virtual ~DepthEstimator() = default;

    // Input: BGR or grayscale image (any size)
    // Output: CV_32FC1 depth map at the same resolution as input, in meters
    virtual cv::Mat estimate(const cv::Mat& image) = 0;

    // Whether the output is metric (absolute scale)
    virtual bool isMetric() const { return false; }
};

}
