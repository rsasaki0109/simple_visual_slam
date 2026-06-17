#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include "io/euroc_dataset.h"
#include "io/euroc_pinhole_calibration.h"

using namespace svslam;

namespace {

// Build a temp path that is unique across concurrent ctest workers. Previously
// relied on std::rand() (deterministic, seed-1 in every new process), which
// caused intermittent collisions when ctest -j launched the same binary in
// parallel. The PID + a process-local counter + a nanosecond timestamp slice
// is more than sufficient for gtest's handful of fixture paths.
std::filesystem::path make_temp_path(const std::string& suffix) {
    static std::atomic<uint64_t> counter{0};
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    const auto id = counter.fetch_add(1);
    std::string name = "svslam_euroc_test_" + std::to_string(getpid()) + "_" +
                       std::to_string(id) + "_" + std::to_string(now_ns) +
                       suffix;
    return std::filesystem::temp_directory_path() / name;
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
  },
  "baseline": 0.110074
})");

    EurocPinholeCalibration calib;
    std::string err;
    ASSERT_TRUE(EurocPinholeCalibration::load_json_file(path.string(), calib, err)) << err;
    EXPECT_DOUBLE_EQ(calib.cam0.fx, 458.654);
    ASSERT_TRUE(calib.has_cam1);
    EXPECT_DOUBLE_EQ(calib.cam1.cx, 379.999);
    ASSERT_EQ(calib.cam1.distortion.size(), 4u);
    EXPECT_TRUE(calib.has_baseline);
    EXPECT_DOUBLE_EQ(calib.baseline_meters, 0.110074);

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

TEST(EurocDatasetTest, LoadsImuDataWhenPresent) {
    const auto root = make_temp_path("");
    const auto cam0_data_dir = root / "mav0" / "cam0" / "data";
    const auto imu_dir = root / "mav0" / "imu0";
    std::filesystem::create_directories(cam0_data_dir);
    std::filesystem::create_directories(imu_dir);

    write_text_file(root / "mav0" / "cam0" / "sensor.yaml",
                    "intrinsics: [50.0, 50.0, 4.0, 3.0]\n"
                    "resolution: [8, 6]\n");
    write_text_file(root / "mav0" / "cam0" / "data.csv",
                    "#timestamp [ns],filename\n"
                    "1000000000,1000000000.png\n");
    write_gray_png(cam0_data_dir / "1000000000.png", 8, 6, 17);

    write_text_file(imu_dir / "data.csv",
                    "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n"
                    "500000000,0.1,0.2,0.3,1.0,2.0,9.81\n"
                    "1500000000,0.11,0.21,0.31,1.01,2.01,9.82\n"
                    "2500000000,0.12,0.22,0.32,1.02,2.02,9.83\n");

    EurocDataset dataset(root.string());
    ASSERT_TRUE(dataset.isValid()) << dataset.error();
    ASSERT_TRUE(dataset.hasImu());
    ASSERT_EQ(dataset.allImu().size(), 3u);

    const auto& first = dataset.allImu().front();
    EXPECT_DOUBLE_EQ(first.timestamp_sec, 0.5);
    EXPECT_DOUBLE_EQ(first.gyro.x(), 0.1);
    EXPECT_DOUBLE_EQ(first.accel.z(), 9.81);

    const auto between = dataset.getImuBetween(1.0, 2.0);
    ASSERT_EQ(between.size(), 1u);
    EXPECT_DOUBLE_EQ(between.front().timestamp_sec, 1.5);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(EurocDatasetTest, ParsesCam0FromImuExtrinsicFromSensorYaml) {
    const auto root = make_temp_path("");
    const auto cam0_data_dir = root / "mav0" / "cam0" / "data";
    std::filesystem::create_directories(cam0_data_dir);

    // EuRoC MH_01's actual cam0 T_BS (body=IMU): ~5 cm forward + 6 cm left,
    // approximately 90 deg rotation about the IMU's X axis.
    write_text_file(root / "mav0" / "cam0" / "sensor.yaml",
                    "intrinsics: [50.0, 50.0, 4.0, 3.0]\n"
                    "resolution: [8, 6]\n"
                    "T_BS:\n"
                    "  cols: 4\n"
                    "  rows: 4\n"
                    "  data: [0.0148655429818, -0.999880929698, 0.00414029679422, "
                    "-0.0216401454975, 0.999557249008, 0.0149672133247, "
                    "0.025715529948, -0.064676986768, -0.0257744366974, "
                    "0.00375618835797, 0.999660727178, 0.00981073058949, "
                    "0.0, 0.0, 0.0, 1.0]\n");
    write_text_file(root / "mav0" / "cam0" / "data.csv",
                    "#timestamp [ns],filename\n"
                    "1000000000,1000000000.png\n");
    write_gray_png(cam0_data_dir / "1000000000.png", 8, 6, 17);

    EurocDataset dataset(root.string());
    ASSERT_TRUE(dataset.isValid()) << dataset.error();
    ASSERT_TRUE(dataset.hasCam0FromImuExtrinsic());

    const SE3 T_cam_imu = dataset.cam0FromImuExtrinsic();
    const Vec3 translation = T_cam_imu.translation();
    EXPECT_NEAR(translation.x(), -0.0216401454975, 1e-9);
    EXPECT_NEAR(translation.y(), -0.064676986768, 1e-9);
    EXPECT_NEAR(translation.z(), 0.00981073058949, 1e-9);

    // Rotation is re-orthonormalized via SVD; check it is still a proper SO(3).
    const Mat33 R = T_cam_imu.rotationMatrix();
    EXPECT_NEAR((R * R.transpose() - Mat33::Identity()).norm(), 0.0, 1e-9);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-9);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(EurocDatasetTest, ParsesMultiLineTBsArray) {
    // Real EuRoC sensor.yaml wraps the 16 T_BS values across four lines.
    const auto root = make_temp_path("");
    const auto cam0_data_dir = root / "mav0" / "cam0" / "data";
    std::filesystem::create_directories(cam0_data_dir);

    write_text_file(root / "mav0" / "cam0" / "sensor.yaml",
                    "intrinsics: [50.0, 50.0, 4.0, 3.0]\n"
                    "resolution: [8, 6]\n"
                    "T_BS:\n"
                    "  cols: 4\n"
                    "  rows: 4\n"
                    "  data: [0.01486, -0.99988, 0.00414, -0.02164,\n"
                    "         0.99956, 0.01497, 0.02572, -0.06468,\n"
                    "        -0.02577, 0.00376, 0.99966, 0.00981,\n"
                    "         0.0, 0.0, 0.0, 1.0]\n");
    write_text_file(root / "mav0" / "cam0" / "data.csv",
                    "#timestamp [ns],filename\n"
                    "1000000000,1000000000.png\n");
    write_gray_png(cam0_data_dir / "1000000000.png", 8, 6, 17);

    EurocDataset dataset(root.string());
    ASSERT_TRUE(dataset.isValid()) << dataset.error();
    ASSERT_TRUE(dataset.hasCam0FromImuExtrinsic());
    const Vec3 t = dataset.cam0FromImuExtrinsic().translation();
    EXPECT_NEAR(t.x(), -0.02164, 1e-5);
    EXPECT_NEAR(t.y(), -0.06468, 1e-5);
    EXPECT_NEAR(t.z(), 0.00981, 1e-5);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(EurocDatasetTest, SilentlySkipsMissingImu) {
    const auto root = make_temp_path("");
    const auto cam0_data_dir = root / "mav0" / "cam0" / "data";
    std::filesystem::create_directories(cam0_data_dir);

    write_text_file(root / "mav0" / "cam0" / "sensor.yaml",
                    "intrinsics: [50.0, 50.0, 4.0, 3.0]\n"
                    "resolution: [8, 6]\n");
    write_text_file(root / "mav0" / "cam0" / "data.csv",
                    "#timestamp [ns],filename\n"
                    "1000000000,1000000000.png\n");
    write_gray_png(cam0_data_dir / "1000000000.png", 8, 6, 17);

    EurocDataset dataset(root.string());
    ASSERT_TRUE(dataset.isValid()) << dataset.error();
    EXPECT_FALSE(dataset.hasImu());
    EXPECT_TRUE(dataset.allImu().empty());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
