#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <filesystem>
#include "core/frame.h"
#include "core/camera.h"
#include "core/map.h"
#include "io/euroc_dataset.h"
#include "io/euroc_pinhole_calibration.h"
#include "io/tum_dataset.h"
#include "io/tum_pinhole_calibration.h"
#include "io/map_io.h"
#include "backend/optimizer.h"
#include "core/heuristic_reference_keyframe_policy.h"
#include "tracking/tracking.h"
#include "backend/local_mapping.h"
#include "loop_closing/loop_closing.h"
#include "depth/stereo_depth_estimator.h"
#ifdef USE_DEPTH_DL
#include "depth/metric_depth_estimator.h"
#include "depth/onnx_depth_estimator.h"
#endif
#include <memory>
#include <stdexcept>
#include <thread>

#include "svslam_version.h"

using namespace svslam;

namespace {

constexpr const char* kRunSummarySchema = "svslam.run_summary.v1";

void print_help(std::ostream& os) {
    os << "SimpleVisualSLAM " << SVSLAM_VERSION_STRING << " - run_mono\n\n"
       << "USAGE\n"
       << "  run_mono --version | -V\n"
       << "  run_mono --help | -h\n"
       << "  run_mono <video_path> [ORBvocab.txt]\n"
       << "  run_mono --euroc <sequence_dir> [options] [ORBvocab.txt]\n"
       << "  run_mono --tum <sequence_dir> [options] [ORBvocab.txt]\n"
       << "\n"
       << "EUROC OPTIONS\n"
       << "  --euroc-camera-config <calib.json> Override cam0/cam1 pinhole intrinsics (+optional distortion)\n"
       << "  --stereo                            Load cam0+cam1 and compute metric stereo depth; tracking still uses cam0\n"
       << "\n"
       << "TUM OPTIONS\n"
       << "  --tum-camera-config <calib.json>   Pinhole intrinsics (+optional distortion); else fr1 defaults\n"
       << "  --depth                             Use depth.txt / sensor depth when available\n"
       << "  --accel                             Load accelerometer.txt into tracker when available\n"
       << "  --repro-eval                        Synchronous mapping; deterministic BA ordering for replay\n"
       << "  --reference-policy heuristic\n"
       << "  --skip-frames N   --max-frames N\n"
       << "  --depth-model <model.onnx>          DL depth (build with -DUSE_DEPTH_DL=ON)\n"
       << "  --metric-depth-model <model.onnx>   Metric DL depth in meters (build with -DUSE_DEPTH_DL=ON)\n"
       << "  --no-viz                            No OpenCV imshow window\n"
       << "  --run-summary-json <path>          Machine-readable run stats (see schema in source)\n"
       << "  --strict-exit                       Exit 3 if tracking did not finish in OK state\n"
       << "\n"
       << "ORB vocabulary: last positional argument, or search data/ORBvoc.txt then ORBvoc.txt\n";
}

bool write_run_summary_json(const std::string& path,
                            int final_tracking_state,
                            int processed_frames,
                            int skipped_frames,
                            std::size_t keyframe_count,
                            std::size_t landmark_count,
                            const TrackingRunStatistics& st,
                            bool map_saved) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to open --run-summary-json: " << path << std::endl;
        return false;
    }
    out << "{\"schema\":\"" << kRunSummarySchema << "\","
        << "\"version\":\"" << SVSLAM_VERSION_STRING << "\","
        << "\"final_tracking_state\":" << final_tracking_state << ','
        << "\"processed_frames\":" << processed_frames << ','
        << "\"skipped_frames\":" << skipped_frames << ','
        << "\"keyframes\":" << keyframe_count << ','
        << "\"landmarks\":" << landmark_count << ','
        << "\"reloc_attempts\":" << st.reloc_attempts << ','
        << "\"reloc_successes\":" << st.reloc_successes << ','
        << "\"frames_tracking_lost\":" << st.frames_tracking_lost << ','
        << "\"reinit_successes\":" << st.reinit_successes << ','
        << "\"map_saved\":" << (map_saved ? "true" : "false") << "}\n";
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string(argv[1]) == "--version" || std::string(argv[1]) == "-V")) {
        std::cout << "SimpleVisualSLAM " << SVSLAM_VERSION_STRING << std::endl;
        return 0;
    }
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_help(std::cout);
        return 0;
    }

    if (argc < 2) {
        std::cerr << "Usage: run_mono --help   (full options)\n"
                  << "  ./run_mono --version | -V     (print semver from CMake project VERSION)\n"
                  << "  ./run_mono <video_path> [vocab_path]\n"
                  << "  ./run_mono --euroc <sequence_dir> [--euroc-camera-config <calib.json>] [--stereo] [vocab_path]\n"
                  << "  ./run_mono --tum <sequence_dir> [--tum-camera-config <calib.json>] [--depth] [--accel] [--repro-eval] [--reference-policy heuristic] [--skip-frames N] [--max-frames N] [--depth-model <path.onnx>] [--metric-depth-model <path.onnx>] [--run-summary-json <path>] [--strict-exit] [vocab_path]\n" << std::endl;
        return -1;
    }

    bool use_euroc = false;
    bool use_tum = false;
    bool use_depth = false;
    bool use_accel = false;
    bool no_viz = false;
    bool repro_eval = false;
    int max_frames = -1;
    int skip_frames = 0;
    std::string reference_policy_name = "heuristic";
    std::string depth_model_path;
    std::string metric_depth_model_path;
    std::string euroc_camera_config;
    std::string tum_camera_config;
    std::string run_summary_json_path;
    bool stereo_mode = false;
    bool strict_exit = false;
    std::string euroc_seq_dir;
    std::string tum_seq_dir;
    std::string input_path;

    // Parse arguments
    int positional_idx = 1;
    if (std::string(argv[1]) == "--euroc") {
        if (argc < 3) {
            std::cerr << "Usage: ./run_mono --euroc <sequence_dir> [vocab_path]" << std::endl;
            return -1;
        }
        use_euroc = true;
        euroc_seq_dir = argv[2];
        positional_idx = 3;
    } else if (std::string(argv[1]) == "--tum") {
        if (argc < 3) {
            std::cerr << "Usage: ./run_mono --tum <sequence_dir> [--depth] [--accel] [vocab_path]" << std::endl;
            return -1;
        }
        use_tum = true;
        tum_seq_dir = argv[2];
        positional_idx = 3;
    } else {
        input_path = argv[1];
        positional_idx = 2;
    }

    auto parse_non_negative_int = [](const std::string& value, const std::string& flag_name) {
        try {
            size_t parsed = 0;
            int result = std::stoi(value, &parsed);
            if (parsed != value.size() || result < 0) {
                throw std::invalid_argument("invalid");
            }
            return result;
        } catch (const std::exception&) {
            throw std::runtime_error(flag_name + " requires a non-negative integer: " + value);
        }
    };

    auto create_reference_policy = [](const std::string& name) -> std::unique_ptr<ReferenceKeyframePolicy> {
        if (name == "heuristic") {
            return std::make_unique<HeuristicReferenceKeyframePolicy>();
        }
        throw std::runtime_error(
            "Unknown reference policy: " + name + " (expected heuristic)");
    };

    // Parse optional flags
    try {
        for (int i = positional_idx; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--depth") {
                use_depth = true;
            } else if (arg == "--accel") {
                use_accel = true;
            } else if (arg == "--repro-eval") {
                repro_eval = true;
            } else if (arg == "--no-viz") {
                no_viz = true;
            } else if (arg == "--reference-policy" && i + 1 < argc) {
                reference_policy_name = argv[++i];
            } else if (arg == "--skip-frames" && i + 1 < argc) {
                skip_frames = parse_non_negative_int(argv[++i], "--skip-frames");
            } else if (arg == "--max-frames" && i + 1 < argc) {
                max_frames = parse_non_negative_int(argv[++i], "--max-frames");
            } else if (arg == "--depth-model" && i + 1 < argc) {
                depth_model_path = argv[++i];
            } else if (arg == "--metric-depth-model" && i + 1 < argc) {
                metric_depth_model_path = argv[++i];
            } else if (arg == "--euroc-camera-config" && i + 1 < argc) {
                euroc_camera_config = argv[++i];
            } else if (arg == "--tum-camera-config" && i + 1 < argc) {
                tum_camera_config = argv[++i];
            } else if (arg == "--run-summary-json" && i + 1 < argc) {
                run_summary_json_path = argv[++i];
            } else if (arg == "--stereo") {
                stereo_mode = true;
            } else if (arg == "--strict-exit") {
                strict_exit = true;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    if (!depth_model_path.empty() && !metric_depth_model_path.empty()) {
        std::cerr << "Specify only one of --depth-model or --metric-depth-model" << std::endl;
        return -1;
    }
    if (stereo_mode && !use_euroc) {
        std::cerr << "--stereo is currently only supported with --euroc" << std::endl;
        return -1;
    }

    // OpenCV RANSAC (solvePnPRansac, findFundamentalMat, findHomography, ...) uses a process-global RNG.
    // Pin the seed so default (async) runs are less run-to-run noisy on the tracking thread. Bitwise replay
    // of full BA state still requires `--repro-eval` (synchronous mapping + deterministic BA ordering).
    cv::setRNGSeed(0);

    cv::VideoCapture cap;
    EurocDataset euroc(".");
    TumRgbdDataset tum(".");

    if (!use_euroc && !use_tum) {
        // Try opening as video
        cap.open(input_path);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video: " << input_path << std::endl;
            return -1;
        }
    } else {
        if (use_euroc) {
            if (!euroc_camera_config.empty()) {
                EurocPinholeCalibration cal;
                std::string cal_err;
                if (!EurocPinholeCalibration::load_json_file(euroc_camera_config, cal, cal_err)) {
                    std::cerr << "Failed to load --euroc-camera-config: " << cal_err << std::endl;
                    return -1;
                }
                euroc = EurocDataset(euroc_seq_dir, cal, stereo_mode);
            } else {
                euroc = EurocDataset(euroc_seq_dir, stereo_mode);
            }
            if (!euroc.isValid()) {
                std::cerr << "Failed to open EuRoC dataset: " << euroc_seq_dir << "\n"
                          << "Reason: " << euroc.error() << std::endl;
                return -1;
            }
        }
        if (use_tum) {
            if (!tum_camera_config.empty()) {
                TumPinholeCalibration cal;
                std::string cal_err;
                if (!TumPinholeCalibration::load_json_file(tum_camera_config, cal, cal_err)) {
                    std::cerr << "Failed to load --tum-camera-config: " << cal_err << std::endl;
                    return -1;
                }
                tum = TumRgbdDataset(tum_seq_dir, cal);
            } else {
                tum = TumRgbdDataset(tum_seq_dir);
            }
            if (!tum.isValid()) {
                std::cerr << "Failed to open TUM dataset: " << tum_seq_dir << "\n"
                          << "Reason: " << tum.error() << std::endl;
                return -1;
            }
        }
    }

    // Initialize Camera
    Camera::Ptr camera;
    if (use_euroc) {
        const cv::Mat& K = euroc.K();
        camera = std::make_shared<Camera>(
            K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2));
    } else if (use_tum) {
        const cv::Mat& K = tum.K();
        camera = std::make_shared<Camera>(
            K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2));
    } else {
        camera = std::make_shared<Camera>(500, 500, 320, 240); // 640x480
    }

    // Initialize ORB detector
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(2000);

    // Initialize Map
    Map::Ptr map = std::make_shared<Map>();
    
    // Optional: Load map if exists
    // if (MapIO::loadMap("map.bin", map)) {
    //     std::cout << "Loaded map from map.bin" << std::endl;
    // }

    // Initialize Local Mapping
    LocalMapping::Ptr local_mapping = std::make_shared<LocalMapping>(map);
    std::thread local_mapping_thread;
    const bool run_local_mapping_thread = !repro_eval;
    if (run_local_mapping_thread) {
        local_mapping_thread = std::thread(&LocalMapping::run, local_mapping);
    }

    // Initialize Loop Closing
    std::string vocab_path;
    // Find vocab path: last argument that isn't a flag
    for (int i = positional_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--depth-model" || arg == "--metric-depth-model" ||
            arg == "--reference-policy" || arg == "--tum-camera-config" ||
            arg == "--euroc-camera-config" ||
            arg == "--run-summary-json" || arg == "--skip-frames" || arg == "--max-frames") {
            ++i;
            continue;
        }
        if (arg != "--depth" && arg != "--accel" && arg != "--repro-eval" && arg != "--no-viz" &&
            arg != "--strict-exit" && arg != "--stereo") {
            vocab_path = arg;
            break;
        }
    }
    if (vocab_path.empty()) {
        if (std::filesystem::exists("data/ORBvoc.txt")) {
            vocab_path = "data/ORBvoc.txt";
        } else {
            vocab_path = "ORBvoc.txt";
        }
    }
    if (!vocab_path.empty() && !std::filesystem::exists(vocab_path)) {
        std::cerr << "LoopClosing: vocab file not found: " << vocab_path << " (loop closing disabled)" << std::endl;
        vocab_path.clear();
    }
    LoopClosing::Ptr loop_closing;
    std::thread loop_closing_thread;
    const bool run_loop_closing_thread = !repro_eval;
    if (run_loop_closing_thread) {
        loop_closing = std::make_shared<LoopClosing>(map, vocab_path);
    }
    
    local_mapping->setLoopClosing(loop_closing);

    // Initialize Tracking
    Tracking::Ptr tracker = std::make_shared<Tracking>();
    tracker->setMap(map);
    tracker->setLocalMapping(local_mapping);
    tracker->setReferenceKeyframePolicy(create_reference_policy(reference_policy_name));
    const std::weak_ptr<Tracking> tracker_weak = tracker;

    std::cout << "Reference keyframe policy: " << reference_policy_name << std::endl;
    if (repro_eval) {
        std::cout << "Repro eval mode: ENABLED (synchronous local mapping, loop closing disabled)" << std::endl;
    }
    if (skip_frames > 0) {
        std::cout << "Skipping first " << skip_frames << " frames before tracking" << std::endl;
    }
    if (max_frames >= 0) {
        std::cout << "Frame budget: " << max_frames << " tracked frames" << std::endl;
    }
    if (use_euroc && stereo_mode) {
        std::cout << "EuRoC stereo mode: ENABLED (tracking cam0 / metric depth from cam0+cam1)" << std::endl;
    }

    // Register BA completion callback to recompute current frame pose
    local_mapping->on_ba_completed_ = [tracker_weak]() {
        if (auto tracker = tracker_weak.lock()) {
            tracker->onBACompleted();
        }
    };
    if (loop_closing) {
        loop_closing->on_loop_corrected_ = [tracker_weak]() {
            if (auto tracker = tracker_weak.lock()) {
                tracker->onLoopCorrected();
            }
        };
        loop_closing_thread = std::thread(&LoopClosing::run, loop_closing);
    }

    // Depth/accel integration setup
    if (use_tum) {
        if (use_depth && tum.hasDepth()) {
            std::cout << "Depth integration: ENABLED (sensor depth)" << std::endl;
        } else if (use_depth) {
            std::cout << "Depth integration: requested but no depth.txt found, DISABLED" << std::endl;
            use_depth = false;
        }
        if (use_accel && tum.hasAccel()) {
            std::cout << "Accelerometer integration: ENABLED" << std::endl;
            tracker->accel_buffer_ = tum.allAccel();
        } else if (use_accel) {
            std::cout << "Accelerometer integration: requested but no accelerometer.txt found, DISABLED" << std::endl;
            use_accel = false;
        }
    }

    if (use_euroc) {
        if (use_accel && euroc.hasImu()) {
            // Mirror the IMU accel channel into accel_buffer_ so the existing
            // gravity-alignment + stationary-detection paths work unchanged,
            // and retain the full IMU (accel + gyro) for future VIO use.
            tracker->imu_buffer_ = euroc.allImu();
            tracker->accel_buffer_.clear();
            tracker->accel_buffer_.reserve(tracker->imu_buffer_.size());
            for (const auto& imu : tracker->imu_buffer_) {
                AccelEntry accel;
                accel.timestamp_sec = imu.timestamp_sec;
                accel.ax = imu.accel.x();
                accel.ay = imu.accel.y();
                accel.az = imu.accel.z();
                tracker->accel_buffer_.push_back(accel);
            }
            if (euroc.hasCam0FromImuExtrinsic()) {
                tracker->setImuToCameraExtrinsic(euroc.cam0FromImuExtrinsic());
            }
            // VIO Stage 0c.e: keep the IMU preintegration residual on by
            // default so sequences where VI init rejects (e.g. MH_01 with
            // noisy early-mono rotations) still benefit from the 9-DoF
            // residual + BA bias blocks — the regression we feared from
            // "preint on with bias=0" never materialises on MH/V1 runs in
            // practice (loose anchor + RW priors soak up the initial
            // mis-estimate). Set SVSLAM_VIO_GATE_PREINT=1 to opt back in to
            // gating preint until VI init succeeds (useful when diagnosing
            // scale-sensitive sequences).
            if (std::getenv("SVSLAM_VIO_GATE_PREINT")) {
                Optimizer::setPreintegrationResidualEnabled(false);
            }
            std::cout << "IMU integration: ENABLED (accel+gyro, "
                      << tracker->imu_buffer_.size() << " samples)" << std::endl;
        } else if (use_accel) {
            std::cout << "IMU integration: requested but mav0/imu0/data.csv absent, DISABLED"
                      << std::endl;
            use_accel = false;
        }
    }

    // Deep learning depth estimator
    if (loop_closing && use_depth) {
        loop_closing->setMetricDepth(true);
    }
    std::shared_ptr<DepthEstimator> stereo_depth_estimator;
    if (use_euroc && stereo_mode) {
        const double stereo_baseline_meters = euroc.stereoBaselineMeters();
        if (stereo_baseline_meters <= 0.0) {
            std::cerr << "EuRoC stereo depth requires a positive stereo baseline from sensor.yaml or "
                         "--euroc-camera-config baseline"
                      << std::endl;
            return -1;
        }
        stereo_depth_estimator = std::make_shared<StereoDepthEstimator>(stereo_baseline_meters, euroc.K());
        std::cout << "Stereo depth estimation: ENABLED (metric, baseline=" << stereo_baseline_meters << " m)"
                  << std::endl;
        if (loop_closing && stereo_depth_estimator->isMetric()) {
            loop_closing->setMetricDepth(true);
        }
    }
#ifdef USE_DEPTH_DL
    std::shared_ptr<DepthEstimator> dl_depth_estimator;
    if (!metric_depth_model_path.empty()) {
        std::cout << "Loading metric DL depth model: " << metric_depth_model_path << std::endl;
        dl_depth_estimator = std::make_shared<MetricDepthEstimator>(metric_depth_model_path, camera);
        std::cout << "DL depth estimation: ENABLED (metric)" << std::endl;
    } else if (!depth_model_path.empty()) {
        std::cout << "Loading DL depth model: " << depth_model_path << std::endl;
        dl_depth_estimator = std::make_shared<OnnxDepthEstimator>(depth_model_path);
        std::cout << "DL depth estimation: ENABLED (relative)" << std::endl;
    }
    if (loop_closing && dl_depth_estimator && dl_depth_estimator->isMetric()) {
        loop_closing->setMetricDepth(true);
    }
#else
    if (!depth_model_path.empty() || !metric_depth_model_path.empty()) {
        std::cerr << "DL depth requires a build with -DUSE_DEPTH_DL=ON" << std::endl;
        return -1;
    }
#endif

    // Trajectory storage (TUM format: timestamp tx ty tz qx qy qz qw)
    struct TrajEntry { double ts, x, y, z, qx, qy, qz, qw; };
    std::vector<TrajEntry> trajectory;

    auto save_online_trajectory = [&](const std::string& path) {
        std::ofstream traj_file(path);
        if (!traj_file.is_open()) return false;
        traj_file << "# timestamp tx ty tz qx qy qz qw\n";
        for (const auto& e : trajectory) {
            traj_file << std::fixed << std::setprecision(6) << e.ts << " "
                      << std::setprecision(9) << e.x << " " << e.y << " " << e.z << " "
                      << e.qx << " " << e.qy << " " << e.qz << " " << e.qw << "\n";
        }
        return true;
    };

    auto save_keyframe_trajectory = [&](const std::string& path) {
        std::ofstream traj_file(path);
        if (!traj_file.is_open()) return false;

        std::vector<Keyframe::Ptr> keyframes;
        keyframes.reserve(map->getAllKeyframes().size());
        for (const auto& kv : map->getAllKeyframes()) {
            if (kv.second) keyframes.push_back(kv.second);
        }

        std::sort(keyframes.begin(), keyframes.end(),
                  [](const Keyframe::Ptr& a, const Keyframe::Ptr& b) {
                      if (a->timestamp_ == b->timestamp_) return a->id_ < b->id_;
                      return a->timestamp_ < b->timestamp_;
                  });

        traj_file << "# timestamp x y z qx qy qz qw\n";
        for (const auto& kf : keyframes) {
            SE3 T_wc = kf->T_cw_.inverse();
            Eigen::Vector3d pos = T_wc.translation();
            Eigen::Quaterniond q = T_wc.unit_quaternion();
            traj_file << std::fixed << std::setprecision(6) << kf->timestamp_ << " "
                      << std::setprecision(9) << pos.x() << " " << pos.y() << " " << pos.z() << " "
                      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        return true;
    };

    // Main Loop
    cv::Mat img;
    cv::Mat right_img;
    cv::Mat depth_img;
    unsigned long frame_id = 0;
    int skipped_frames = 0;
    int processed_frames = 0;
    while (true) {
        if (max_frames >= 0 && processed_frames >= max_frames) {
            break;
        }

        double timestamp = 0.0;
        depth_img = cv::Mat();
        right_img = cv::Mat();
        if (!use_euroc && !use_tum) {
            cap >> img;
            if (img.empty()) break;
            timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        } else {
            if (use_euroc) {
                if (stereo_mode) {
                    if (!euroc.next(img, right_img, timestamp)) break;
                } else {
                    if (!euroc.next(img, timestamp)) break;
                }
            } else if (use_depth) {
                if (!tum.nextWithDepth(img, depth_img, timestamp)) break;
            } else {
                if (!tum.next(img, timestamp)) break;
            }
        }

        if (skipped_frames < skip_frames) {
            skipped_frames++;
            continue;
        }

        // Create Frame
        Frame::Ptr frame = std::make_shared<Frame>(frame_id++, timestamp, camera, img);

        // Attach depth if available
        if (!depth_img.empty()) {
            frame->depth_image_ = depth_img;
            frame->depth_is_metric_ = true;
        }
        else if (stereo_depth_estimator && !right_img.empty()) {
            frame->depth_image_ = stereo_depth_estimator->estimate(img, right_img);
            frame->depth_is_metric_ = stereo_depth_estimator->isMetric();
        }
#ifdef USE_DEPTH_DL
        else if (dl_depth_estimator) {
            // Only run DL depth every N frames to reduce CPU cost
            // Always run on first frame (init) and every 5th frame (keyframe candidates)
            bool run_dl = (frame_id <= 1) || (frame_id % 5 == 0);
            if (run_dl) {
                frame->depth_image_ = dl_depth_estimator->estimate(img);
                frame->depth_is_metric_ = dl_depth_estimator->isMetric();
                frame->depth_is_learned_ = true;
            }
        }
#endif

        // Extract Features
        frame->extractORB(orb);

        // Track
        tracker->addFrame(frame);
        if (repro_eval) {
            local_mapping->processPendingWork();
        }

        // Save trajectory (camera position in world frame, TUM format)
        SE3 T_wc = frame->getPose().inverse();
        Eigen::Vector3d pos = T_wc.translation();
        Eigen::Quaterniond q = T_wc.unit_quaternion();
        trajectory.push_back({timestamp, pos.x(), pos.y(), pos.z(), q.x(), q.y(), q.z(), q.w()});

        std::cout << "Frame " << frame->id_
                  << ": " << frame->keypoints_.size() << " kps"
                  << " | State: " << (int)tracker->state_
                  << " | Pose: " << pos.transpose()
                  << std::endl;
        processed_frames++;

        // Visualization
        if (!no_viz) {
            cv::Mat img_show;
            cv::drawKeypoints(img, frame->keypoints_, img_show);
            cv::putText(img_show, "State: " + std::to_string((int)tracker->state_), cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            cv::Mat viz_frame = img_show;
            if (stereo_mode && !right_img.empty()) {
                cv::Mat left_show = img_show.clone();
                cv::Mat right_show;
                cv::cvtColor(right_img, right_show, cv::COLOR_GRAY2BGR);
                cv::putText(left_show, "Left", cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 0), 2);
                cv::putText(right_show, "Right", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 0), 2);
                cv::hconcat(left_show, right_show, viz_frame);
            }

            cv::imshow("SimpleVisualSLAM", viz_frame);
            char k = cv::waitKey(10);
            if (k == 27) break;
            if (frame_id == 100) {
                cv::imwrite("slam_result.jpg", viz_frame);
            }
        }
    }
    
    std::cout << "Finished processing." << std::endl;
    std::cout << "Processed frames: " << processed_frames
              << " (skipped " << skipped_frames << ")" << std::endl;

    if (save_online_trajectory("trajectory.txt")) {
        std::cout << "Trajectory saved to trajectory.txt (" << trajectory.size() << " poses)" << std::endl;
    }
    if (save_online_trajectory("trajectory_online.txt")) {
        std::cout << "Trajectory saved to trajectory_online.txt" << std::endl;
    }

    // Request worker threads to stop before waiting so shutdown does not enqueue more work.
    if (run_local_mapping_thread) {
        local_mapping->requestStop();
        local_mapping_thread.join();
    } else {
        local_mapping->processPendingWork();
    }
    if (run_loop_closing_thread) {
        loop_closing->requestStop();
        loop_closing_thread.join();
    }
    
    // Save Map
    std::cout << "Saving map to map.bin..." << std::endl;
    const bool map_saved_ok = MapIO::saveMap("map.bin", map);
    if (map_saved_ok) {
        std::cout << "Map saved successfully." << std::endl;
    } else {
        std::cerr << "Failed to save map." << std::endl;
    }

    if (save_keyframe_trajectory("trajectory_keyframes.txt")) {
        std::cout << "Keyframe trajectory saved to trajectory_keyframes.txt (" << map->getAllKeyframes().size()
                  << " keyframes)" << std::endl;
    }

    const TrackingRunStatistics tr_stats = tracker->runStatistics();
    if (!run_summary_json_path.empty()) {
        if (!write_run_summary_json(run_summary_json_path, static_cast<int>(tracker->state_), processed_frames,
                                    skipped_frames, map->getAllKeyframes().size(), map->getAllLandmarks().size(),
                                    tr_stats, map_saved_ok)) {
            std::cerr << "Run summary was not written." << std::endl;
        } else {
            std::cout << "Run summary written to " << run_summary_json_path << std::endl;
        }
    }

    // Plan comments for future steps
    /*
     * Development Plan:
     * 
     * 1. Tracking:
     *    - Implement 'TrackReferenceKeyframe': Match features with previous keyframe.
     *    - Implement 'TrackLocalMap': Project local map points to current frame and optimize pose.
     *    - Implement Motion Model: Initialize pose from previous frame velocity.
     *
     * 2. Initialization:
     *    - Implement Monocular Initialization (Homography/Fundamental matrix).
     *    - Triangulate initial MapPoints.
     *    - Create initial Keyframes and Map.
     *
     * 3. Local Mapping (Backend):
     *    - 'ProcessNewKeyframe': Add new KF to map.
     *    - 'MapPointCulling': Remove bad points.
     *    - 'CreateNewMapPoints': Triangulate new points from connected KFs.
     *    - 'LocalBundleAdjustment': Optimize local KFs and MPs using Ceres.
     *
     * 4. Loop Closure:
     *    - Integrate DBoW2.
     *    - 'DetectLoop': Query BoW database.
     *    - 'ComputeSim3': Geometric verification.
     *    - 'CorrectLoop': Pose Graph Optimization using Ceres/g2o (or implement custom in Ceres).
     *
     * 5. Persistence (Map Save/Load):
     *    - Implement full serialization in MapIO.
     *    - Serialize Camera, Keyframes (Pose, Features), Landmarks (Pos, Descriptors), Graph (Weights).
     */

    if (strict_exit && tracker->state_ != TrackingState::OK) {
        return 3;
    }
    return 0;
}
