#include <iostream>
#include <string>
#include <vector>
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
                  << "  ./run_mono --tum <sequence_dir> [vocab_path]\n" << std::endl;
        return -1;
    }

    bool use_euroc = false;
    bool use_tum = false;
    std::string euroc_seq_dir;
    std::string tum_seq_dir;
    std::string input_path;

    if (std::string(argv[1]) == "--euroc") {
        if (argc < 3) {
            std::cerr << "Usage: ./run_mono --euroc <sequence_dir> [vocab_path]" << std::endl;
            return -1;
        }
        use_euroc = true;
        euroc_seq_dir = argv[2];
    } else if (std::string(argv[1]) == "--tum") {
        if (argc < 3) {
            std::cerr << "Usage: ./run_mono --tum <sequence_dir> [vocab_path]" << std::endl;
            return -1;
        }
        use_tum = true;
        tum_seq_dir = argv[2];
    } else {
        input_path = argv[1];
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
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(1000);

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
    int vocab_arg_index = 2;
    if (use_euroc || use_tum) vocab_arg_index = 3;

    if (argc >= vocab_arg_index + 1) {
        vocab_path = argv[vocab_arg_index];
    } else {
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
    std::thread loop_closing_thread(&LoopClosing::run, loop_closing);
    
    // Connect LocalMapping to LoopClosing (Keyframes should be passed to LoopClosing)
    // We need to add a method to LocalMapping to set LoopClosing
    local_mapping->setLoopClosing(loop_closing);

    // Initialize Tracking
    Tracking::Ptr tracker = std::make_shared<Tracking>();
    tracker->setMap(map);
    tracker->setLocalMapping(local_mapping);


    // Main Loop
    cv::Mat img;
    unsigned long frame_id = 0;
    while (true) {
        double timestamp = 0.0;
        if (!use_euroc && !use_tum) {
            cap >> img;
            if (img.empty()) break;
            timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        } else {
            if (use_euroc) {
                if (!euroc.next(img, timestamp)) break;
            } else {
                if (!tum.next(img, timestamp)) break;
            }
        }

        // Create Frame
        Frame::Ptr frame = std::make_shared<Frame>(frame_id++, timestamp, camera, img);

        // Extract Features
        frame->extractORB(orb);

        // Track
        tracker->addFrame(frame);

        std::cout << "Frame " << frame->id_ 
                  << ": " << frame->keypoints_.size() << " kps"
                  << " | State: " << (int)tracker->state_ 
                  << " | Pose: " << frame->getPose().translation().transpose() 
                  << std::endl;

        // Visualization
        cv::Mat img_show;
        cv::drawKeypoints(img, frame->keypoints_, img_show);
        
        // Draw pose info
        cv::putText(img_show, "State: " + std::to_string((int)tracker->state_), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        cv::imshow("SimpleVisualSLAM", img_show);
        char k = cv::waitKey(10);
        if (k == 27) break;
        
        // Save the 100th frame as a sample result
        if (frame_id == 100) {
            cv::imwrite("slam_result.jpg", img_show);
            std::cout << "Saved slam_result.jpg" << std::endl;
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
