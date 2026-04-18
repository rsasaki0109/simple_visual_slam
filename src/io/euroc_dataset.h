#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "core/common.h"
#include "io/euroc_pinhole_calibration.h"
#include "sensors/imu.h"

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

    // IMU accessors (empty vector if mav0/imu0/data.csv is absent).
    bool hasImu() const { return !imu_entries_.empty(); }
    const std::vector<ImuEntry>& allImu() const { return imu_entries_; }
    // Returns IMU samples strictly within (t0, t1]. Inputs in seconds.
    std::vector<ImuEntry> getImuBetween(double t0, double t1) const;

    // Extrinsic for cam0: T_cam_imu (transforms IMU-body-frame points to
    // cam0 frame). EuRoC sensor.yaml stores this as T_BS with body := IMU.
    // Returns identity if cam0.T_BS was not parsed.
    SE3 cam0FromImuExtrinsic() const { return cam0_from_imu_; }
    bool hasCam0FromImuExtrinsic() const { return has_cam0_from_imu_; }

private:
    struct CsvEntry {
        long long timestamp_ns;
        double timestamp_sec;
        std::string image_path;
    };

    EurocDataset(const std::string& seq_dir, const EurocPinholeCalibration* calib, bool load_stereo);

    bool loadSensorYaml(const std::string& sensor_yaml_path, EurocPinholeCalibration::Camera& calib);
    bool loadDataCsv(const std::string& data_csv_path, const std::string& data_dir, std::vector<CsvEntry>& entries);
    // Populates imu_entries_ from mav0/imu0/data.csv. Returns true on success
    // or if file is absent (IMU is optional). False only on parse error.
    bool loadImuCsv(const std::string& imu_csv_path);
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
    std::vector<ImuEntry> imu_entries_;
    size_t index_ = 0;
    bool stereo_enabled_ = false;
    double stereo_baseline_meters_ = 0.0;

    SE3 cam0_from_imu_;  // defaults to identity
    bool has_cam0_from_imu_ = false;
};

}  // namespace svslam
