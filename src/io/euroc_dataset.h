#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "io/euroc_pinhole_calibration.h"

namespace svslam {

class EurocDataset {
public:
    struct Entry {
        double timestamp_sec;
        std::string left_image_path;
        std::string right_image_path;
    };

    explicit EurocDataset(const std::string& seq_dir);
    EurocDataset(const std::string& seq_dir, bool load_stereo);
    EurocDataset(const std::string& seq_dir, const EurocPinholeCalibration& calib, bool load_stereo = false);

    bool isValid() const;
    const std::string& error() const;

    const cv::Mat& K() const;
    const cv::Mat& rightK() const;
    double stereoBaselineMeters() const;
    bool hasStereo() const;

    bool next(cv::Mat& image, double& timestamp_sec);
    bool next(cv::Mat& left_image, cv::Mat& right_image, double& timestamp_sec);

private:
    struct CsvEntry {
        long long timestamp_ns;
        double timestamp_sec;
        std::string image_path;
    };

    EurocDataset(const std::string& seq_dir, const EurocPinholeCalibration* calib, bool load_stereo);

    bool loadSensorYaml(const std::string& sensor_yaml_path, EurocPinholeCalibration::Camera& calib);
    bool loadDataCsv(const std::string& data_csv_path, const std::string& data_dir, std::vector<CsvEntry>& entries);
    bool buildStereoEntries(const std::vector<CsvEntry>& left_entries, const std::vector<CsvEntry>& right_entries);
    void buildMonoEntries(const std::vector<CsvEntry>& left_entries);
    void initCalibration(const EurocPinholeCalibration::Camera& calib,
                         cv::Mat& K,
                         cv::Mat& dist_coeffs,
                         cv::Mat& new_K,
                         cv::Mat& undist_map1,
                         cv::Mat& undist_map2);
    bool loadImage(const std::string& path,
                   const cv::Mat& undist_map1,
                   const cv::Mat& undist_map2,
                   cv::Mat& image);

    std::string seq_dir_;
    std::string error_;

    cv::Mat K_;
    cv::Mat right_K_;
    cv::Mat dist_coeffs_;
    cv::Mat right_dist_coeffs_;
    cv::Mat new_K_;
    cv::Mat right_new_K_;
    cv::Mat undist_map1_;
    cv::Mat undist_map2_;
    cv::Mat right_undist_map1_;
    cv::Mat right_undist_map2_;

    std::vector<Entry> entries_;
    size_t index_ = 0;
    bool stereo_enabled_ = false;
    double stereo_baseline_meters_ = 0.0;
};

}  // namespace svslam
