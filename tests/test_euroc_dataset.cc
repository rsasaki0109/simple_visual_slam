#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "io/euroc_dataset.h"
#include "io/euroc_pinhole_calibration.h"

using namespace svslam;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    return std::filesystem::temp_directory_path() /
           ("svslam_euroc_test_" + std::to_string(std::rand()) + suffix);
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream ofs(path);
    ofs << content;
}

void write_gray_png(const std::filesystem::path& path, int width, int height, int pixel_value) {
    const cv::Mat image(height, width, CV_8UC1, cv::Scalar(pixel_value));
    ASSERT_TRUE(cv::imwrite(path.string(), image));
}

}  // namespace

TEST(EurocPinholeCalibrationTest, LoadsStereoJsonConfig) {
    const auto path = make_temp_path(".json");
    write_text_file(path, R"({
  "cam0": {
    "fx": 458.654,
    "fy": 457.296,
    "cx": 367.215,
    "cy": 248.375,
    "width": 752,
    "height": 480,
    "distortion": [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
  },
  "cam1": {
    "fx": 457.587,
    "fy": 456.134,
    "cx": 379.999,
    "cy": 255.238,
    "width": 752,
    "height": 480,
    "distortion": [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]
  }
})");

    EurocPinholeCalibration calib;
    std::string err;
    ASSERT_TRUE(EurocPinholeCalibration::load_json_file(path.string(), calib, err)) << err;
    EXPECT_DOUBLE_EQ(calib.cam0.fx, 458.654);
    ASSERT_TRUE(calib.has_cam1);
    EXPECT_DOUBLE_EQ(calib.cam1.cx, 379.999);
    ASSERT_EQ(calib.cam1.distortion.size(), 4u);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(EurocDatasetTest, LoadsStereoImagePairs) {
    const auto root = make_temp_path("");
    const auto cam0_data_dir = root / "mav0" / "cam0" / "data";
    const auto cam1_data_dir = root / "mav0" / "cam1" / "data";
    std::filesystem::create_directories(cam0_data_dir);
    std::filesystem::create_directories(cam1_data_dir);

    write_text_file(root / "mav0" / "cam0" / "sensor.yaml",
                    "intrinsics: [50.0, 50.0, 4.0, 3.0]\n"
                    "resolution: [8, 6]\n");
    write_text_file(root / "mav0" / "cam1" / "sensor.yaml",
                    "intrinsics: [51.0, 51.0, 4.0, 3.0]\n"
                    "resolution: [8, 6]\n");

    write_text_file(root / "mav0" / "cam0" / "data.csv",
                    "#timestamp [ns],filename\n"
                    "1000000000,1000000000.png\n"
                    "2000000000,2000000000.png\n");
    write_text_file(root / "mav0" / "cam1" / "data.csv",
                    "#timestamp [ns],filename\n"
                    "1000000000,1000000000.png\n"
                    "2000000000,2000000000.png\n");

    write_gray_png(cam0_data_dir / "1000000000.png", 8, 6, 17);
    write_gray_png(cam1_data_dir / "1000000000.png", 8, 6, 93);
    write_gray_png(cam0_data_dir / "2000000000.png", 8, 6, 25);
    write_gray_png(cam1_data_dir / "2000000000.png", 8, 6, 101);

    EurocDataset dataset(root.string(), true);
    ASSERT_TRUE(dataset.isValid()) << dataset.error();
    EXPECT_TRUE(dataset.hasStereo());
    EXPECT_FALSE(dataset.rightK().empty());

    cv::Mat left;
    cv::Mat right;
    double timestamp_sec = 0.0;
    ASSERT_TRUE(dataset.next(left, right, timestamp_sec)) << dataset.error();
    EXPECT_DOUBLE_EQ(timestamp_sec, 1.0);
    ASSERT_FALSE(left.empty());
    ASSERT_FALSE(right.empty());
    EXPECT_EQ(left.rows, 6);
    EXPECT_EQ(left.cols, 8);
    EXPECT_EQ(left.at<unsigned char>(0, 0), 17);
    EXPECT_EQ(right.at<unsigned char>(0, 0), 93);

    ASSERT_TRUE(dataset.next(left, right, timestamp_sec)) << dataset.error();
    EXPECT_DOUBLE_EQ(timestamp_sec, 2.0);
    EXPECT_EQ(left.at<unsigned char>(0, 0), 25);
    EXPECT_EQ(right.at<unsigned char>(0, 0), 101);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
