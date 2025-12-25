#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "core/frame.h"
#include "core/camera.h"
#include "core/map.h"
#include "io/map_io.h"
#include "tracking/tracking.h"
#include "backend/local_mapping.h"
#include <thread>

using namespace svslam;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./run_mono <video_path_or_image_folder>" << std::endl;
        return -1;
    }

    std::string input_path = argv[1];
    cv::VideoCapture cap;
    
    // Try opening as video
    cap.open(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << input_path << std::endl;
        return -1;
    }

    // Initialize Camera
    Camera::Ptr camera = std::make_shared<Camera>(500, 500, 320, 240); // 640x480

    // Initialize ORB detector
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(1000);

    // Initialize Map
    Map::Ptr map = std::make_shared<Map>();

    // Initialize Local Mapping
    LocalMapping::Ptr local_mapping = std::make_shared<LocalMapping>(map);
    std::thread local_mapping_thread(&LocalMapping::run, local_mapping);

    // Initialize Tracking
    Tracking::Ptr tracker = std::make_shared<Tracking>();
    tracker->setMap(map);
    tracker->setLocalMapping(local_mapping);

    // Main Loop
    cv::Mat img;
    unsigned long frame_id = 0;
    while (true) {
        cap >> img;
        if (img.empty()) break;

        // Create Frame
        double timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
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
    }
    
    std::cout << "Finished processing." << std::endl;
    
    // Stop Local Mapping
    local_mapping->requestStop();
    local_mapping_thread.join();

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
