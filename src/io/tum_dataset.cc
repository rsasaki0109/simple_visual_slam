#include "io/tum_dataset.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

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

}  // namespace

TumRgbdDataset::TumRgbdDataset(const std::string& seq_dir) : seq_dir_(seq_dir) {
    const std::string rgb_txt = (std::filesystem::path(seq_dir_) / "rgb.txt").string();
    const std::string rgb_dir = (std::filesystem::path(seq_dir_) / "rgb").string();

    if (!std::filesystem::exists(rgb_txt)) {
        error_ = "rgb.txt not found: " + rgb_txt;
        return;
    }
    if (!std::filesystem::exists(rgb_dir)) {
        error_ = "rgb dir not found: " + rgb_dir;
        return;
    }

    // TUM RGB-D fr1 camera intrinsics (RGB camera)
    K_ = cv::Mat::eye(3, 3, CV_64F);
    K_.at<double>(0, 0) = 517.3;
    K_.at<double>(1, 1) = 516.5;
    K_.at<double>(0, 2) = 318.6;
    K_.at<double>(1, 2) = 255.3;

    dist_coeffs_ = (cv::Mat_<double>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);

    // Pre-compute undistortion maps for efficiency
    cv::Size img_size(640, 480);  // TUM fr1 image size
    new_K_ = cv::getOptimalNewCameraMatrix(K_, dist_coeffs_, img_size, 0, img_size);
    cv::initUndistortRectifyMap(K_, dist_coeffs_, cv::Mat(), new_K_, img_size, CV_32FC1, undist_map1_, undist_map2_);
    // Use undistorted intrinsics going forward
    K_ = new_K_.clone();

    if (!loadRgbTxt(rgb_txt, rgb_dir)) return;
    if (entries_.empty()) {
        error_ = "no entries in rgb.txt";
        return;
    }

    // Try to load depth.txt (optional)
    const std::string depth_txt = (std::filesystem::path(seq_dir_) / "depth.txt").string();
    if (std::filesystem::exists(depth_txt)) {
        loadDepthTxt(depth_txt);
    }

    // Try to load accelerometer.txt (optional)
    const std::string accel_txt = (std::filesystem::path(seq_dir_) / "accelerometer.txt").string();
    if (std::filesystem::exists(accel_txt)) {
        loadAccelerometerTxt(accel_txt);
    }
}

bool TumRgbdDataset::isValid() const { return error_.empty(); }

const std::string& TumRgbdDataset::error() const { return error_; }

const cv::Mat& TumRgbdDataset::K() const { return K_; }

bool TumRgbdDataset::next(cv::Mat& image, double& timestamp_sec) {
    if (!isValid()) return false;
    if (index_ >= entries_.size()) return false;

    const auto& e = entries_[index_++];
    timestamp_sec = e.timestamp_sec;

    image = cv::imread(e.image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        error_ = "failed to read image: " + e.image_path;
        return false;
    }

    // Apply lens undistortion
    if (!undist_map1_.empty()) {
        cv::Mat undistorted;
        cv::remap(image, undistorted, undist_map1_, undist_map2_, cv::INTER_LINEAR);
        image = undistorted;
    }

    return true;
}

bool TumRgbdDataset::nextWithDepth(cv::Mat& rgb, cv::Mat& depth, double& timestamp_sec) {
    if (!isValid()) return false;
    if (index_ >= entries_.size()) return false;

    const auto& e = entries_[index_];
    // Don't increment index_ here - call next() to do that
    double ts = e.timestamp_sec;

    // Find associated depth
    int depth_idx = findNearestDepth(ts);

    // Read RGB through normal path
    if (!next(rgb, timestamp_sec)) return false;

    // Load depth if available
    depth = cv::Mat();
    if (depth_idx >= 0) {
        const auto& de = depth_entries_[depth_idx];
        depth = cv::imread(de.depth_path, cv::IMREAD_UNCHANGED);
        if (!depth.empty() && depth.type() != CV_16UC1) {
            depth = cv::Mat();  // Invalid format
        }
    }

    return true;
}

std::vector<AccelEntry> TumRgbdDataset::getAccelBetween(double t0, double t1) const {
    std::vector<AccelEntry> result;
    for (const auto& a : accel_entries_) {
        if (a.timestamp_sec >= t0 && a.timestamp_sec <= t1) {
            result.push_back(a);
        }
    }
    return result;
}

int TumRgbdDataset::findNearestDepth(double rgb_timestamp, double max_diff_sec) const {
    if (depth_entries_.empty()) return -1;

    int best_idx = -1;
    double best_diff = max_diff_sec;

    // Binary search for approximate location, then linear scan
    auto it = std::lower_bound(depth_entries_.begin(), depth_entries_.end(), rgb_timestamp,
        [](const DepthEntry& entry, double ts) { return entry.timestamp_sec < ts; });

    // Check a few entries around the found position
    int start = std::max(0, static_cast<int>(std::distance(depth_entries_.begin(), it)) - 2);
    int end = std::min(static_cast<int>(depth_entries_.size()), start + 5);

    for (int i = start; i < end; ++i) {
        double diff = std::abs(depth_entries_[i].timestamp_sec - rgb_timestamp);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }

    return best_idx;
}

bool TumRgbdDataset::loadRgbTxt(const std::string& rgb_txt_path, const std::string& rgb_dir) {
    std::ifstream ifs(rgb_txt_path);
    if (!ifs.is_open()) {
        error_ = "failed to open rgb.txt: " + rgb_txt_path;
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::stringstream ss(line);
        std::string ts_str;
        std::string rel;
        if (!(ss >> ts_str >> rel)) continue;

        const double ts = std::stod(ts_str);
        std::string img_path = (std::filesystem::path(seq_dir_) / rel).string();
        if (!std::filesystem::exists(img_path)) {
            img_path = (std::filesystem::path(rgb_dir) / rel).string();
        }
        if (!std::filesystem::exists(img_path)) continue;

        entries_.push_back({ts, img_path});
    }

    if (entries_.empty()) {
        error_ = "no readable image entries from rgb.txt: " + rgb_txt_path;
        return false;
    }

    return true;
}

bool TumRgbdDataset::loadDepthTxt(const std::string& depth_txt_path) {
    std::ifstream ifs(depth_txt_path);
    if (!ifs.is_open()) return false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string ts_str, rel;
        if (!(ss >> ts_str >> rel)) continue;

        double ts = std::stod(ts_str);
        std::string depth_path = (std::filesystem::path(seq_dir_) / rel).string();
        if (!std::filesystem::exists(depth_path)) continue;

        depth_entries_.push_back({ts, depth_path});
    }

    if (!depth_entries_.empty()) {
        std::cout << "TUM: Loaded " << depth_entries_.size() << " depth entries" << std::endl;
    }
    return !depth_entries_.empty();
}

bool TumRgbdDataset::loadAccelerometerTxt(const std::string& accel_txt_path) {
    std::ifstream ifs(accel_txt_path);
    if (!ifs.is_open()) return false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string ts_str;
        double ax, ay, az;
        if (!(ss >> ts_str >> ax >> ay >> az)) continue;

        double ts = std::stod(ts_str);
        accel_entries_.push_back({ts, ax, ay, az});
    }

    if (!accel_entries_.empty()) {
        std::cout << "TUM: Loaded " << accel_entries_.size() << " accelerometer entries" << std::endl;
    }
    return !accel_entries_.empty();
}

}  // namespace svslam
