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
#include "io/tum_dataset.h"
#include "io/map_io.h"
#include "tracking/tracking.h"
#include "backend/local_mapping.h"
#include "loop_closing/loop_closing.h"
#include <thread>

using namespace svslam;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  ./run_mono <video_path> [vocab_path]\n"
                  << "  ./run_mono --euroc <sequence_dir> [vocab_path]\n"
                  << "  ./run_mono --tum <sequence_dir> [--depth] [--accel] [vocab_path]\n" << std::endl;
        return -1;
    }

    bool use_euroc = false;
    bool use_tum = false;
    bool use_depth = false;
    bool use_accel = false;
    bool no_viz = false;
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

    // Parse optional flags
    for (int i = positional_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--depth") {
            use_depth = true;
        } else if (arg == "--accel") {
            use_accel = true;
        } else if (arg == "--no-viz") {
            no_viz = true;
        }
    }

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
            euroc = EurocDataset(euroc_seq_dir);
            if (!euroc.isValid()) {
                std::cerr << "Failed to open EuRoC dataset: " << euroc_seq_dir << "\n"
                          << "Reason: " << euroc.error() << std::endl;
                return -1;
            }
        }
        if (use_tum) {
            tum = TumRgbdDataset(tum_seq_dir);
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
    std::thread local_mapping_thread(&LocalMapping::run, local_mapping);

    // Initialize Loop Closing
    std::string vocab_path;
    // Find vocab path: last argument that isn't a flag
    for (int i = positional_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg != "--depth" && arg != "--accel" && arg != "--no-viz") {
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
    LoopClosing::Ptr loop_closing = std::make_shared<LoopClosing>(map, vocab_path);
    if (use_depth) {
        loop_closing->setMetricDepth(true);
    }
    std::thread loop_closing_thread(&LoopClosing::run, loop_closing);
    
    // Connect LocalMapping to LoopClosing (Keyframes should be passed to LoopClosing)
    // We need to add a method to LocalMapping to set LoopClosing
    local_mapping->setLoopClosing(loop_closing);

    // Initialize Tracking
    Tracking::Ptr tracker = std::make_shared<Tracking>();
    tracker->setMap(map);
    tracker->setLocalMapping(local_mapping);

    // Register BA completion callback to recompute current frame pose
    local_mapping->on_ba_completed_ = [tracker]() {
        tracker->onBACompleted();
    };

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

    // Trajectory storage (TUM format: timestamp tx ty tz qx qy qz qw)
    struct TrajEntry { double ts, x, y, z, qx, qy, qz, qw; };
    std::vector<TrajEntry> trajectory;

    // Main Loop
    cv::Mat img;
    cv::Mat depth_img;
    unsigned long frame_id = 0;
    while (true) {
        double timestamp = 0.0;
        depth_img = cv::Mat();
        if (!use_euroc && !use_tum) {
            cap >> img;
            if (img.empty()) break;
            timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        } else {
            if (use_euroc) {
                if (!euroc.next(img, timestamp)) break;
            } else if (use_depth) {
                if (!tum.nextWithDepth(img, depth_img, timestamp)) break;
            } else {
                if (!tum.next(img, timestamp)) break;
            }
        }

        // Create Frame
        Frame::Ptr frame = std::make_shared<Frame>(frame_id++, timestamp, camera, img);

        // Attach depth if available
        if (!depth_img.empty()) {
            frame->depth_image_ = depth_img;
            frame->depth_is_metric_ = true;
        }

        // Extract Features
        frame->extractORB(orb);

        // Track
        tracker->addFrame(frame);

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

        // Visualization
        if (!no_viz) {
            cv::Mat img_show;
            cv::drawKeypoints(img, frame->keypoints_, img_show);
            cv::putText(img_show, "State: " + std::to_string((int)tracker->state_), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            cv::imshow("SimpleVisualSLAM", img_show);
            char k = cv::waitKey(10);
            if (k == 27) break;
            if (frame_id == 100) {
                cv::imwrite("slam_result.jpg", img_show);
            }
        }
    }
    
    std::cout << "Finished processing." << std::endl;
    
    // Stop Local Mapping
    local_mapping->requestStop();
    local_mapping_thread.join();
    
    // Stop Loop Closing
    loop_closing->requestStop();
    loop_closing_thread.join();
    
    // Save Map
    std::cout << "Saving map to map.bin..." << std::endl;
    if (MapIO::saveMap("map.bin", map)) {
        std::cout << "Map saved successfully." << std::endl;
    } else {
        std::cerr << "Failed to save map." << std::endl;
    }

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

    if (save_online_trajectory("trajectory.txt")) {
        std::cout << "Trajectory saved to trajectory.txt (" << trajectory.size() << " poses)" << std::endl;
    }
    if (save_online_trajectory("trajectory_online.txt")) {
        std::cout << "Trajectory saved to trajectory_online.txt" << std::endl;
    }
    if (save_keyframe_trajectory("trajectory_keyframes.txt")) {
        std::cout << "Keyframe trajectory saved to trajectory_keyframes.txt (" << map->getAllKeyframes().size()
                  << " keyframes)" << std::endl;
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

    return 0;
}
