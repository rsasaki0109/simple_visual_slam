#include "io/euroc_pinhole_calibration.h"

#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>

namespace svslam {

namespace {

bool read_file(const std::string& path, std::string& out, std::string& error) {
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

bool parse_double(const std::string& json, const std::string& key, double& value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return false;
    }
    value = std::stod(m[1].str());
    return true;
}

bool parse_int(const std::string& json, const std::string& key, int& value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return false;
    }
    value = std::stoi(m[1].str());
    return true;
}

bool parse_distortion_array(const std::string& json, std::vector<double>& coeffs) {
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

bool extract_object(const std::string& json, const std::string& key, std::string& out) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    const size_t colon_pos = json.find(':', key_pos + needle.size());
    if (colon_pos == std::string::npos) {
        return false;
    }

    const size_t brace_pos = json.find('{', colon_pos);
    if (brace_pos == std::string::npos) {
        return false;
    }

    int depth = 0;
    for (size_t i = brace_pos; i < json.size(); ++i) {
        if (json[i] == '{') {
            ++depth;
        } else if (json[i] == '}') {
            --depth;
            if (depth == 0) {
                out = json.substr(brace_pos, i - brace_pos + 1);
                return true;
            }
        }
    }

    return false;
}

bool parse_camera(const std::string& json, EurocPinholeCalibration::Camera& out, std::string& error) {
    EurocPinholeCalibration::Camera c;
    if (!parse_double(json, "fx", c.fx) || !parse_double(json, "fy", c.fy) || !parse_double(json, "cx", c.cx) ||
        !parse_double(json, "cy", c.cy)) {
        error = "JSON camera must contain numeric keys: fx, fy, cx, cy";
        return false;
    }

    int w = 0;
    int h = 0;
    if (parse_int(json, "width", w) && w > 0) {
        c.image_width = w;
    }
    if (parse_int(json, "height", h) && h > 0) {
        c.image_height = h;
    }
    if (!parse_distortion_array(json, c.distortion)) {
        error = "invalid distortion array";
        return false;
    }

    out = c;
    error.clear();
    return true;
}

}  // namespace

bool EurocPinholeCalibration::load_json_file(const std::string& path,
                                             EurocPinholeCalibration& out,
                                             std::string& error) {
    std::string json;
    if (!read_file(path, json, error)) {
        return false;
    }

    EurocPinholeCalibration calib;
    std::string cam0_json;
    if (extract_object(json, "cam0", cam0_json)) {
        if (!parse_camera(cam0_json, calib.cam0, error)) {
            error = "invalid cam0 calibration in " + path + ": " + error;
            return false;
        }
    } else {
        if (!parse_camera(json, calib.cam0, error)) {
            error += " (" + path + ")";
            return false;
        }
    }

    std::string cam1_json;
    if (extract_object(json, "cam1", cam1_json)) {
        if (!parse_camera(cam1_json, calib.cam1, error)) {
            error = "invalid cam1 calibration in " + path + ": " + error;
            return false;
        }
        calib.has_cam1 = true;
    }

    double baseline_meters = 0.0;
    if (parse_double(json, "baseline", baseline_meters) && baseline_meters > 0.0) {
        calib.baseline_meters = baseline_meters;
        calib.has_baseline = true;
    }

    out = calib;
    error.clear();
    return true;
}

}  // namespace svslam
