#include "io/euroc_dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
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

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

}  // namespace

EurocDataset::EurocDataset(const std::string& seq_dir) : seq_dir_(seq_dir) {
    const std::string cam0_dir = (std::filesystem::path(seq_dir_) / "mav0" / "cam0").string();
    const std::string sensor_yaml = (std::filesystem::path(cam0_dir) / "sensor.yaml").string();
    const std::string data_csv = (std::filesystem::path(cam0_dir) / "data.csv").string();
    const std::string data_dir = (std::filesystem::path(cam0_dir) / "data").string();

    if (!std::filesystem::exists(sensor_yaml)) {
        error_ = "sensor.yaml not found: " + sensor_yaml;
        return;
    }
    if (!std::filesystem::exists(data_csv)) {
        error_ = "data.csv not found: " + data_csv;
        return;
    }

    if (!loadSensorYaml(sensor_yaml)) return;
    if (!loadDataCsv(data_csv, data_dir)) return;

    if (entries_.empty()) {
        error_ = "no entries in data.csv";
        return;
    }
}

bool EurocDataset::isValid() const { return error_.empty(); }

const std::string& EurocDataset::error() const { return error_; }

const cv::Mat& EurocDataset::K() const { return K_; }

bool EurocDataset::next(cv::Mat& image, double& timestamp_sec) {
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

bool EurocDataset::loadSensorYaml(const std::string& sensor_yaml_path) {
    std::ifstream ifs(sensor_yaml_path);
    if (!ifs.is_open()) {
        error_ = "failed to open sensor.yaml: " + sensor_yaml_path;
        return false;
    }

    // Minimal parser for EuRoC sensor.yaml
    // Example:
    // intrinsics: [458.654, 457.296, 367.215, 248.375]
    // resolution: [752, 480]
    double fx = 0, fy = 0, cx = 0, cy = 0;
    bool got_intrinsics = false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (startsWith(line, "intrinsics:")) {
            auto pos = line.find('[');
            auto pos2 = line.find(']');
            if (pos == std::string::npos || pos2 == std::string::npos || pos2 <= pos) continue;
            const std::string body = line.substr(pos + 1, pos2 - pos - 1);

            std::vector<double> vals;
            std::stringstream ss(body);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                tok = trim(tok);
                if (tok.empty()) continue;
                vals.push_back(std::stod(tok));
            }

            if (vals.size() == 4) {
                fx = vals[0];
                fy = vals[1];
                cx = vals[2];
                cy = vals[3];
                got_intrinsics = true;
                break;
            }
        }
    }

    if (!got_intrinsics) {
        error_ = "failed to parse intrinsics from sensor.yaml: " + sensor_yaml_path;
        return false;
    }

    K_ = cv::Mat::eye(3, 3, CV_64F);
    K_.at<double>(0, 0) = fx;
    K_.at<double>(1, 1) = fy;
    K_.at<double>(0, 2) = cx;
    K_.at<double>(1, 2) = cy;

    return true;
}

bool EurocDataset::loadDataCsv(const std::string& data_csv_path, const std::string& data_dir) {
    std::ifstream ifs(data_csv_path);
    if (!ifs.is_open()) {
        error_ = "failed to open data.csv: " + data_csv_path;
        return false;
    }

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
                entries_.push_back({ts_sec, img_path_png});
                continue;
            }
            continue;
        }

        entries_.push_back({ts_sec, img_path});
    }

    if (entries_.empty()) {
        error_ = "no readable image entries from data.csv: " + data_csv_path;
        return false;
    }

    return true;
}

}  // namespace svslam
