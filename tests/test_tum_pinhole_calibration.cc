#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "io/tum_pinhole_calibration.h"

using namespace svslam;

namespace {

std::filesystem::path make_temp_json(const std::string& content) {
    auto dir = std::filesystem::temp_directory_path();
    auto path = dir / ("svslam_calib_test_" + std::to_string(std::rand()) + ".json");
    std::ofstream ofs(path);
    ofs << content;
    return path;
}

}  // namespace

TEST(TumPinholeCalibrationTest, LoadsMinimalJsonWithoutDistortion) {
    const auto path = make_temp_json(R"({"fx": 500.5, "fy": 501.25, "cx": 320, "cy": 240})");
    TumPinholeCalibration c;
    std::string err;
    ASSERT_TRUE(TumPinholeCalibration::load_json_file(path.string(), c, err)) << err;
    EXPECT_DOUBLE_EQ(c.fx, 500.5);
    EXPECT_DOUBLE_EQ(c.fy, 501.25);
    EXPECT_DOUBLE_EQ(c.cx, 320.0);
    EXPECT_DOUBLE_EQ(c.cy, 240.0);
    EXPECT_TRUE(c.distortion.empty());
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(TumPinholeCalibrationTest, Fr1DefaultMatchesHardcodedTum) {
    const TumPinholeCalibration c = TumPinholeCalibration::fr1_default();
    EXPECT_DOUBLE_EQ(c.fx, 517.3);
    EXPECT_EQ(c.distortion.size(), 5u);
}

TEST(TumPinholeCalibrationTest, RejectsMissingFx) {
    const auto path = make_temp_json(R"({"fy": 1, "cx": 2, "cy": 3})");
    TumPinholeCalibration c;
    std::string err;
    EXPECT_FALSE(TumPinholeCalibration::load_json_file(path.string(), c, err));
    EXPECT_FALSE(err.empty());
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
