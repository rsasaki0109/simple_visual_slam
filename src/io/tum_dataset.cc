#include "io/tum_dataset.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

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
    // ref: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    // fr1: fx=517.3 fy=516.5 cx=318.6 cy=255.3
    K_ = cv::Mat::eye(3, 3, CV_64F);
    K_.at<double>(0, 0) = 517.3;
    K_.at<double>(1, 1) = 516.5;
    K_.at<double>(0, 2) = 318.6;
    K_.at<double>(1, 2) = 255.3;

    if (!loadRgbTxt(rgb_txt, rgb_dir)) return;
    if (entries_.empty()) {
        error_ = "no entries in rgb.txt";
        return;
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

    return true;
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

}  // namespace svslam
