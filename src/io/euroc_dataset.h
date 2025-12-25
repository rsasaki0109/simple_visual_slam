#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace svslam {

class EurocDataset {
public:
    struct Entry {
        double timestamp_sec;
        std::string image_path;
    };

    explicit EurocDataset(const std::string& seq_dir);

    bool isValid() const;
    const std::string& error() const;

    const cv::Mat& K() const;

    bool next(cv::Mat& image, double& timestamp_sec);

private:
    bool loadSensorYaml(const std::string& sensor_yaml_path);
    bool loadDataCsv(const std::string& data_csv_path, const std::string& data_dir);

    std::string seq_dir_;
    std::string error_;

    cv::Mat K_;

    std::vector<Entry> entries_;
    size_t index_ = 0;
};

}
