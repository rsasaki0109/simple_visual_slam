#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace svslam {

class TumRgbdDataset {
public:
    struct Entry {
        double timestamp_sec;
        std::string image_path;
    };

    explicit TumRgbdDataset(const std::string& seq_dir);

    bool isValid() const;
    const std::string& error() const;

    const cv::Mat& K() const;

    bool next(cv::Mat& image, double& timestamp_sec);

private:
    bool loadRgbTxt(const std::string& rgb_txt_path, const std::string& rgb_dir);

    std::string seq_dir_;
    std::string error_;

    cv::Mat K_;

    std::vector<Entry> entries_;
    size_t index_ = 0;
};

}
