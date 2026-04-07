#include "io/tum_pinhole_calibration.h"

#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>

namespace svslam {

TumPinholeCalibration TumPinholeCalibration::fr1_default() {
    TumPinholeCalibration c;
    c.fx = 517.3;
    c.fy = 516.5;
    c.cx = 318.6;
    c.cy = 255.3;
    c.image_width = 640;
    c.image_height = 480;
    c.distortion = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
    return c;
}

namespace {

static bool read_file(const std::string& path, std::string& out, std::string& error) {
    std::ifstream ifs(path);
    if (!ifs) {
        error = "cannot open file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

static bool parse_double(const std::string& json, const std::string& key, double& value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return false;
    }
    value = std::stod(m[1].str());
    return true;
}

static bool parse_int(const std::string& json, const std::string& key, int& value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return false;
    }
    value = std::stoi(m[1].str());
    return true;
}

static bool parse_distortion_array(const std::string& json, std::vector<double>& coeffs) {
    coeffs.clear();
    const std::regex re("\"distortion\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return true;  // optional
    }
    std::string inner = m[1].str();
    std::string token;
    for (char ch : inner) {
        if (ch == ',' || ch == '\n' || ch == '\r') {
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
                token.erase(token.begin());
            }
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
                token.pop_back();
            }
            if (!token.empty()) {
                coeffs.push_back(std::stod(token));
            }
            token.clear();
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) {
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
            token.erase(token.begin());
        }
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
            token.pop_back();
        }
        if (!token.empty()) {
            coeffs.push_back(std::stod(token));
        }
    }
    return true;
}

}  // namespace

bool TumPinholeCalibration::load_json_file(const std::string& path, TumPinholeCalibration& out, std::string& error) {
    std::string json;
    if (!read_file(path, json, error)) {
        return false;
    }
    TumPinholeCalibration c;
    if (!parse_double(json, "fx", c.fx) || !parse_double(json, "fy", c.fy) || !parse_double(json, "cx", c.cx) ||
        !parse_double(json, "cy", c.cy)) {
        error = "JSON must contain numeric keys: fx, fy, cx, cy (" + path + ")";
        return false;
    }
    int w = 0, h = 0;
    if (parse_int(json, "width", w) && w > 0) {
        c.image_width = w;
    }
    if (parse_int(json, "height", h) && h > 0) {
        c.image_height = h;
    }
    if (!parse_distortion_array(json, c.distortion)) {
        error = "invalid distortion array in " + path;
        return false;
    }
    out = c;
    error.clear();
    return true;
}

}  // namespace svslam
