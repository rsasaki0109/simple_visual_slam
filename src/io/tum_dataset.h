#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace svslam {

struct AccelEntry {
    double timestamp_sec;
    double ax, ay, az;
};

class TumRgbdDataset {
public:
    struct Entry {
        double timestamp_sec;
        std::string image_path;
    };

    struct DepthEntry {
        double timestamp_sec;
        std::string depth_path;
    };

    explicit TumRgbdDataset(const std::string& seq_dir);

    bool isValid() const;
    const std::string& error() const;

    const cv::Mat& K() const;

    bool next(cv::Mat& image, double& timestamp_sec);
    bool nextWithDepth(cv::Mat& rgb, cv::Mat& depth, double& timestamp_sec);

    std::vector<AccelEntry> getAccelBetween(double t0, double t1) const;
    const std::vector<AccelEntry>& allAccel() const { return accel_entries_; }

    bool hasDepth() const { return !depth_entries_.empty(); }
    bool hasAccel() const { return !accel_entries_.empty(); }

private:
    bool loadRgbTxt(const std::string& rgb_txt_path, const std::string& rgb_dir);
    bool loadDepthTxt(const std::string& depth_txt_path);
    bool loadAccelerometerTxt(const std::string& accel_txt_path);
    int findNearestDepth(double rgb_timestamp, double max_diff_sec = 0.03) const;

    std::string seq_dir_;
    std::string error_;

    cv::Mat K_;
    cv::Mat dist_coeffs_;
    cv::Mat new_K_;
    cv::Mat undist_map1_, undist_map2_;

    std::vector<Entry> entries_;
    std::vector<DepthEntry> depth_entries_;
    std::vector<AccelEntry> accel_entries_;
    size_t index_ = 0;
};

}
