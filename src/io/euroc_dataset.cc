#include "io/euroc_dataset.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace svslam {

namespace {

static std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
    return s.substr(b, e - b);
}

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

static bool parseArrayLine(const std::string& line, const std::string& key, std::vector<double>& values) {
    if (!startsWith(line, key)) {
        return false;
    }

    const auto pos = line.find('[');
    const auto pos2 = line.find(']');
    if (pos == std::string::npos || pos2 == std::string::npos || pos2 <= pos) {
        return false;
    }

    values.clear();
    const std::string body = line.substr(pos + 1, pos2 - pos - 1);
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (tok.empty()) continue;
        values.push_back(std::stod(tok));
    }

    return true;
}

static double computeStereoBaselineMeters(const EurocPinholeCalibration::Camera& cam0,
                                          const EurocPinholeCalibration::Camera& cam1) {
    if (cam0.T_BS.size() != 16 || cam1.T_BS.size() != 16) {
        return 0.0;
    }

    const double dx = cam1.T_BS[3] - cam0.T_BS[3];
    const double dy = cam1.T_BS[7] - cam0.T_BS[7];
    const double dz = cam1.T_BS[11] - cam0.T_BS[11];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

EurocDataset::EurocDataset(const std::string& seq_dir) : EurocDataset(seq_dir, nullptr, false) {}

EurocDataset::EurocDataset(const std::string& seq_dir, bool load_stereo) : EurocDataset(seq_dir, nullptr, load_stereo) {}

EurocDataset::EurocDataset(const std::string& seq_dir,
                           const EurocPinholeCalibration& calib,
                           bool load_stereo)
    : EurocDataset(seq_dir, &calib, load_stereo) {}

EurocDataset::EurocDataset(const std::string& seq_dir,
                           const EurocPinholeCalibration* calib,
                           bool load_stereo)
    : seq_dir_(seq_dir), stereo_enabled_(load_stereo) {
    const std::filesystem::path cam0_dir = std::filesystem::path(seq_dir_) / "mav0" / "cam0";
    const std::filesystem::path cam1_dir = std::filesystem::path(seq_dir_) / "mav0" / "cam1";

    const std::string cam0_sensor_yaml = (cam0_dir / "sensor.yaml").string();
    const std::string cam0_data_csv = (cam0_dir / "data.csv").string();
    const std::string cam0_data_dir = (cam0_dir / "data").string();

    const std::string cam1_sensor_yaml = (cam1_dir / "sensor.yaml").string();
    const std::string cam1_data_csv = (cam1_dir / "data.csv").string();
    const std::string cam1_data_dir = (cam1_dir / "data").string();

    if (!std::filesystem::exists(cam0_data_csv)) {
        error_ = "data.csv not found: " + cam0_data_csv;
        return;
    }
    if (stereo_enabled_ && !std::filesystem::exists(cam1_data_csv)) {
        error_ = "cam1 data.csv not found: " + cam1_data_csv;
        return;
    }

    EurocPinholeCalibration::Camera cam0_calib;
    bool have_cam0_calib = false;
    if (std::filesystem::exists(cam0_sensor_yaml)) {
        if (loadSensorYaml(cam0_sensor_yaml, cam0_calib)) {
            have_cam0_calib = true;
        } else if (calib == nullptr) {
            return;
        } else {
            error_.clear();
        }
    }
    if (calib != nullptr) {
        const std::vector<double> sensor_t_bs = cam0_calib.T_BS;
        cam0_calib = calib->cam0;
        if (cam0_calib.T_BS.empty()) {
            cam0_calib.T_BS = sensor_t_bs;
        }
        have_cam0_calib = true;
    }
    if (!have_cam0_calib) {
        error_ = "cam0 sensor.yaml not found and no external EuRoC calibration provided";
        return;
    }

    EurocPinholeCalibration::Camera cam1_calib;
    bool have_cam1_calib = false;
    if (stereo_enabled_ && std::filesystem::exists(cam1_sensor_yaml)) {
        if (loadSensorYaml(cam1_sensor_yaml, cam1_calib)) {
            have_cam1_calib = true;
        } else if (calib == nullptr || !calib->has_cam1) {
            return;
        } else {
            error_.clear();
        }
    }
    if (stereo_enabled_ && calib != nullptr && calib->has_cam1) {
        const std::vector<double> sensor_t_bs = cam1_calib.T_BS;
        cam1_calib = calib->cam1;
        if (cam1_calib.T_BS.empty()) {
            cam1_calib.T_BS = sensor_t_bs;
        }
        have_cam1_calib = true;
    }
    if (stereo_enabled_ && !have_cam1_calib) {
        error_ = "cam1 sensor.yaml not found and no external cam1 calibration provided";
        return;
    }

    initCalibration(cam0_calib, K_, dist_coeffs_, new_K_, undist_map1_, undist_map2_);
    if (stereo_enabled_) {
        initCalibration(cam1_calib, right_K_, right_dist_coeffs_, right_new_K_, right_undist_map1_,
                        right_undist_map2_);
        stereo_baseline_meters_ =
            (calib != nullptr && calib->has_baseline) ? calib->baseline_meters
                                                      : computeStereoBaselineMeters(cam0_calib, cam1_calib);
    }

    std::vector<CsvEntry> left_entries;
    if (!loadDataCsv(cam0_data_csv, cam0_data_dir, left_entries)) return;

    if (stereo_enabled_) {
        std::vector<CsvEntry> right_entries;
        if (!loadDataCsv(cam1_data_csv, cam1_data_dir, right_entries)) return;
        if (!buildStereoEntries(left_entries, right_entries)) return;
    } else {
        buildMonoEntries(left_entries);
    }

    if (entries_.empty()) {
        error_ = stereo_enabled_ ? "no stereo entries in data.csv" : "no entries in data.csv";
        return;
    }

    // IMU is optional — only fail on parse error, not on missing file.
    const std::string imu_csv =
        (std::filesystem::path(seq_dir_) / "mav0" / "imu0" / "data.csv").string();
    if (std::filesystem::exists(imu_csv)) {
        if (!loadImuCsv(imu_csv)) {
            return;
        }
    }
}

bool EurocDataset::isValid() const { return error_.empty(); }

const std::string& EurocDataset::error() const { return error_; }

const cv::Mat& EurocDataset::K() const { return K_; }

const cv::Mat& EurocDataset::rightK() const { return right_K_; }

double EurocDataset::stereoBaselineMeters() const { return stereo_baseline_meters_; }

bool EurocDataset::hasStereo() const {
    return stereo_enabled_ && !entries_.empty() && !entries_.front().right_image_path.empty();
}

bool EurocDataset::next(cv::Mat& image, double& timestamp_sec) {
    if (!isValid()) return false;
    if (index_ >= entries_.size()) return false;

    const auto& e = entries_[index_++];
    timestamp_sec = e.timestamp_sec;

    return loadImage(e.left_image_path, undist_map1_, undist_map2_, image);
}

bool EurocDataset::next(cv::Mat& left_image, cv::Mat& right_image, double& timestamp_sec) {
    if (!isValid()) return false;
    if (!hasStereo()) {
        error_ = "stereo mode not enabled for this EuRoC dataset";
        return false;
    }
    if (index_ >= entries_.size()) return false;

    const auto& e = entries_[index_++];
    timestamp_sec = e.timestamp_sec;

    if (!loadImage(e.left_image_path, undist_map1_, undist_map2_, left_image)) {
        return false;
    }
    return loadImage(e.right_image_path, right_undist_map1_, right_undist_map2_, right_image);
}

bool EurocDataset::loadSensorYaml(const std::string& sensor_yaml_path, EurocPinholeCalibration::Camera& calib) {
    std::ifstream ifs(sensor_yaml_path);
    if (!ifs.is_open()) {
        error_ = "failed to open sensor.yaml: " + sensor_yaml_path;
        return false;
    }

    bool got_intrinsics = false;
    bool got_resolution = false;
    std::vector<double> values;
    calib = EurocPinholeCalibration::Camera();
    bool in_t_bs_block = false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (startsWith(line, "T_BS:")) {
            in_t_bs_block = true;
            continue;
        }
        if (in_t_bs_block && parseArrayLine(line, "data:", values)) {
            if (values.size() == 16) {
                calib.T_BS = values;
            }
            in_t_bs_block = false;
            continue;
        }

        if (parseArrayLine(line, "intrinsics:", values)) {
            if (values.size() >= 4) {
                calib.fx = values[0];
                calib.fy = values[1];
                calib.cx = values[2];
                calib.cy = values[3];
                got_intrinsics = true;
            }
            continue;
        }
        if (parseArrayLine(line, "resolution:", values)) {
            if (values.size() >= 2) {
                calib.image_width = static_cast<int>(values[0]);
                calib.image_height = static_cast<int>(values[1]);
                got_resolution = true;
            }
            continue;
        }
        if (parseArrayLine(line, "distortion_coefficients:", values)) {
            calib.distortion = values;
        }
    }

    if (!got_intrinsics) {
        error_ = "failed to parse intrinsics from sensor.yaml: " + sensor_yaml_path;
        return false;
    }
    if (!got_resolution) {
        calib.image_width = 752;
        calib.image_height = 480;
    }
    return true;
}

bool EurocDataset::loadDataCsv(const std::string& data_csv_path,
                               const std::string& data_dir,
                               std::vector<CsvEntry>& entries) {
    std::ifstream ifs(data_csv_path);
    if (!ifs.is_open()) {
        error_ = "failed to open data.csv: " + data_csv_path;
        return false;
    }

    entries.clear();
    std::string line;
    bool first = true;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (first) {
            // header: #timestamp [ns],filename
            first = false;
            if (line.find("timestamp") != std::string::npos) continue;
        }

        std::stringstream ss(line);
        std::string ts_str;
        std::string fn;
        if (!std::getline(ss, ts_str, ',')) continue;
        if (!std::getline(ss, fn)) continue;

        ts_str = trim(ts_str);
        fn = trim(fn);
        if (ts_str.empty() || fn.empty()) continue;

        const long long ts_ns = std::stoll(ts_str);
        const double ts_sec = static_cast<double>(ts_ns) * 1e-9;
        const std::string img_path = (std::filesystem::path(data_dir) / fn).string();

        if (!std::filesystem::exists(img_path)) {
            // Some datasets store without extension; try adding .png
            const std::string img_path_png = img_path + ".png";
            if (std::filesystem::exists(img_path_png)) {
                entries.push_back({ts_ns, ts_sec, img_path_png});
                continue;
            }
            continue;
        }

        entries.push_back({ts_ns, ts_sec, img_path});
    }

    if (entries.empty()) {
        error_ = "no readable image entries from data.csv: " + data_csv_path;
        return false;
    }

    return true;
}

bool EurocDataset::loadImuCsv(const std::string& imu_csv_path) {
    std::ifstream ifs(imu_csv_path);
    if (!ifs.is_open()) {
        error_ = "failed to open imu0 data.csv: " + imu_csv_path;
        return false;
    }

    imu_entries_.clear();
    std::string line;
    bool first = true;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        if (first) {
            first = false;
            if (line.find("timestamp") != std::string::npos) continue;
        }

        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, ',')) {
            fields.push_back(trim(field));
        }
        // Expect: ts_ns, wx, wy, wz, ax, ay, az
        if (fields.size() < 7) continue;

        try {
            const long long ts_ns = std::stoll(fields[0]);
            ImuEntry e;
            e.timestamp_sec = static_cast<double>(ts_ns) * 1e-9;
            e.gyro  = Vec3(std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3]));
            e.accel = Vec3(std::stod(fields[4]), std::stod(fields[5]), std::stod(fields[6]));
            imu_entries_.push_back(e);
        } catch (const std::exception&) {
            continue;
        }
    }

    std::sort(imu_entries_.begin(), imu_entries_.end(),
              [](const ImuEntry& a, const ImuEntry& b) {
                  return a.timestamp_sec < b.timestamp_sec;
              });
    return true;
}

std::vector<ImuEntry> EurocDataset::getImuBetween(double t0, double t1) const {
    std::vector<ImuEntry> out;
    if (imu_entries_.empty() || !(t1 > t0)) {
        return out;
    }
    // Binary search for first sample with timestamp > t0.
    auto lo = std::upper_bound(
        imu_entries_.begin(), imu_entries_.end(), t0,
        [](double v, const ImuEntry& e) { return v < e.timestamp_sec; });
    for (auto it = lo; it != imu_entries_.end() && it->timestamp_sec <= t1; ++it) {
        out.push_back(*it);
    }
    return out;
}

bool EurocDataset::buildStereoEntries(const std::vector<CsvEntry>& left_entries,
                                      const std::vector<CsvEntry>& right_entries) {
    entries_.clear();

    size_t left_idx = 0;
    size_t right_idx = 0;
    while (left_idx < left_entries.size() && right_idx < right_entries.size()) {
        const long long left_ts = left_entries[left_idx].timestamp_ns;
        const long long right_ts = right_entries[right_idx].timestamp_ns;

        if (left_ts == right_ts) {
            entries_.push_back({left_entries[left_idx].timestamp_sec, left_entries[left_idx].image_path,
                                right_entries[right_idx].image_path});
            ++left_idx;
            ++right_idx;
        } else if (left_ts < right_ts) {
            ++left_idx;
        } else {
            ++right_idx;
        }
    }

    if (entries_.empty()) {
        error_ = "no stereo pairs matched between cam0 and cam1";
        return false;
    }

    return true;
}

void EurocDataset::buildMonoEntries(const std::vector<CsvEntry>& left_entries) {
    entries_.clear();
    entries_.reserve(left_entries.size());
    for (const auto& entry : left_entries) {
        entries_.push_back({entry.timestamp_sec, entry.image_path, ""});
    }
}

void EurocDataset::initCalibration(const EurocPinholeCalibration::Camera& calib,
                                   cv::Mat& K,
                                   cv::Mat& dist_coeffs,
                                   cv::Mat& new_K,
                                   cv::Mat& undist_map1,
                                   cv::Mat& undist_map2) {
    K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = calib.fx;
    K.at<double>(1, 1) = calib.fy;
    K.at<double>(0, 2) = calib.cx;
    K.at<double>(1, 2) = calib.cy;

    if (calib.distortion.empty()) {
        dist_coeffs = cv::Mat();
        new_K = cv::Mat();
        undist_map1.release();
        undist_map2.release();
        return;
    }

    dist_coeffs.create(static_cast<int>(calib.distortion.size()), 1, CV_64F);
    for (size_t i = 0; i < calib.distortion.size(); ++i) {
        dist_coeffs.at<double>(static_cast<int>(i), 0) = calib.distortion[i];
    }

    const cv::Size img_size(calib.image_width, calib.image_height);
    new_K = cv::getOptimalNewCameraMatrix(K, dist_coeffs, img_size, 0, img_size);
    cv::initUndistortRectifyMap(K, dist_coeffs, cv::Mat(), new_K, img_size, CV_32FC1, undist_map1, undist_map2);
    K = new_K.clone();
}

bool EurocDataset::loadImage(const std::string& path,
                             const cv::Mat& undist_map1,
                             const cv::Mat& undist_map2,
                             cv::Mat& image) {
    image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        error_ = "failed to read image: " + path;
        return false;
    }

    if (!undist_map1.empty()) {
        cv::Mat undistorted;
        cv::remap(image, undistorted, undist_map1, undist_map2, cv::INTER_LINEAR);
        image = undistorted;
    }

    return true;
}

}  // namespace svslam
