#include "io/map_io.h"
#include "core/keyframe.h"
#include "core/landmark.h"
#include "core/camera.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/core/eigen.hpp>

namespace svslam {

bool MapIO::saveMap(const std::string& filename, Map::Ptr map) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) return false;
    
    // Magic Header
    std::string magic = "SVSLAM";
    ofs.write(magic.c_str(), magic.size());
    
    // 1. Save Camera (Assuming single camera for all KFs)
    // We get camera from the first keyframe or pass it explicitly.
    // Ideally Map should hold the Camera or we assume strict monocular single cam.
    auto keyframes = map->getAllKeyframes();
    if (keyframes.empty()) {
        std::cout << "MapIO: No keyframes to save." << std::endl;
        return false;
    }
    
    Camera::Ptr camera = keyframes.begin()->second->camera_;
    ofs.write((char*)&camera->fx_, sizeof(double));
    ofs.write((char*)&camera->fy_, sizeof(double));
    ofs.write((char*)&camera->cx_, sizeof(double));
    ofs.write((char*)&camera->cy_, sizeof(double));
    ofs.write((char*)&camera->k1_, sizeof(double));
    ofs.write((char*)&camera->k2_, sizeof(double));
    ofs.write((char*)&camera->p1_, sizeof(double));
    ofs.write((char*)&camera->p2_, sizeof(double));
    ofs.write((char*)&camera->k3_, sizeof(double));
    
    // 2. Save Keyframes
    size_t num_kfs = keyframes.size();
    ofs.write((char*)&num_kfs, sizeof(size_t));
    
    for (auto& pair : keyframes) {
        auto kf = pair.second;
        
        // ID & Timestamp
        ofs.write((char*)&kf->id_, sizeof(unsigned long));
        ofs.write((char*)&kf->timestamp_, sizeof(double));
        
        // Pose (T_cw) - 7 doubles (tx, ty, tz, qx, qy, qz, qw)
        Eigen::Vector3d t = kf->T_cw_.translation();
        Eigen::Quaterniond q = kf->T_cw_.unit_quaternion();
        ofs.write((char*)t.data(), sizeof(double) * 3);
        ofs.write((char*)q.coeffs().data(), sizeof(double) * 4); // coeffs is (x, y, z, w)
        
        // Keypoints
        size_t num_kps = kf->keypoints_.size();
        ofs.write((char*)&num_kps, sizeof(size_t));
        for (const auto& kp : kf->keypoints_) {
            ofs.write((char*)&kp.pt.x, sizeof(float));
            ofs.write((char*)&kp.pt.y, sizeof(float));
            ofs.write((char*)&kp.size, sizeof(float));
            ofs.write((char*)&kp.octave, sizeof(int));
        }
        
        // Descriptors (cv::Mat)
        // Rows = num_kps, Cols = 32 (ORB), Type = CV_8U
        int rows = kf->descriptors_.rows;
        int cols = kf->descriptors_.cols;
        int type = kf->descriptors_.type();
        ofs.write((char*)&rows, sizeof(int));
        ofs.write((char*)&cols, sizeof(int));
        ofs.write((char*)&type, sizeof(int));
        if (rows > 0 && cols > 0) {
            size_t data_size = kf->descriptors_.total() * kf->descriptors_.elemSize();
            ofs.write((char*)kf->descriptors_.data, data_size);
        }
        
        // Landmark Associations
        // Save ID of landmark for each keypoint. -1 if none.
        for (auto& lm : kf->landmarks_) {
            long long lm_id = -1;
            if (lm) {
                lm_id = (long long)lm->id_;
            }
            ofs.write((char*)&lm_id, sizeof(long long));
        }
    }
    
    // 3. Save Landmarks
    auto landmarks = map->getAllLandmarks();
    size_t num_lms = landmarks.size();
    ofs.write((char*)&num_lms, sizeof(size_t));
    
    for (auto& pair : landmarks) {
        auto lm = pair.second;
        
        // ID
        ofs.write((char*)&lm->id_, sizeof(unsigned long));
        
        // Position
        Vec3 pos = lm->getPos();
        ofs.write((char*)pos.data(), sizeof(double) * 3);
        
        // Descriptor
        int rows = lm->descriptor_.rows;
        int cols = lm->descriptor_.cols;
        int type = lm->descriptor_.type();
        ofs.write((char*)&rows, sizeof(int));
        ofs.write((char*)&cols, sizeof(int));
        ofs.write((char*)&type, sizeof(int));
        if (rows > 0 && cols > 0) {
            size_t data_size = lm->descriptor_.total() * lm->descriptor_.elemSize();
            ofs.write((char*)lm->descriptor_.data, data_size);
        }
    }
    
    std::cout << "MapIO: Saved " << num_kfs << " KFs and " << num_lms << " LMs to " << filename << std::endl;
    return true;
}

bool MapIO::loadMap(const std::string& filename, Map::Ptr map) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) return false;
    
    // Magic Header
    char magic[7] = {0};
    ifs.read(magic, 6);
    if (std::string(magic) != "SVSLAM") {
        std::cout << "MapIO: Invalid magic header." << std::endl;
        return false;
    }
    
    // 1. Load Camera
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;
    ifs.read((char*)&fx, sizeof(double));
    ifs.read((char*)&fy, sizeof(double));
    ifs.read((char*)&cx, sizeof(double));
    ifs.read((char*)&cy, sizeof(double));
    ifs.read((char*)&k1, sizeof(double));
    ifs.read((char*)&k2, sizeof(double));
    ifs.read((char*)&p1, sizeof(double));
    ifs.read((char*)&p2, sizeof(double));
    ifs.read((char*)&k3, sizeof(double));
    
    Camera::Ptr camera = std::make_shared<Camera>(fx, fy, cx, cy, k1, k2, p1, p2, k3);
    
    // 2. Load Keyframes
    size_t num_kfs = 0;
    ifs.read((char*)&num_kfs, sizeof(size_t));
    
    std::map<unsigned long, std::vector<long long>> kf_lm_ids; // Store associations
    
    for (size_t i = 0; i < num_kfs; ++i) {
        unsigned long id;
        double timestamp;
        ifs.read((char*)&id, sizeof(unsigned long));
        ifs.read((char*)&timestamp, sizeof(double));
        
        // Pose
        Eigen::Vector3d t;
        Eigen::Quaterniond q;
        ifs.read((char*)t.data(), sizeof(double) * 3);
        ifs.read((char*)q.coeffs().data(), sizeof(double) * 4); // x, y, z, w
        SE3 T_cw(q, t);
        
        // Keypoints
        size_t num_kps = 0;
        ifs.read((char*)&num_kps, sizeof(size_t));
        std::vector<cv::KeyPoint> keypoints(num_kps);
        for (size_t j = 0; j < num_kps; ++j) {
            ifs.read((char*)&keypoints[j].pt.x, sizeof(float));
            ifs.read((char*)&keypoints[j].pt.y, sizeof(float));
            ifs.read((char*)&keypoints[j].size, sizeof(float));
            ifs.read((char*)&keypoints[j].octave, sizeof(int));
        }
        
        // Descriptors
        int rows, cols, type;
        ifs.read((char*)&rows, sizeof(int));
        ifs.read((char*)&cols, sizeof(int));
        ifs.read((char*)&type, sizeof(int));
        cv::Mat descriptors;
        if (rows > 0 && cols > 0) {
            descriptors.create(rows, cols, type);
            size_t data_size = descriptors.total() * descriptors.elemSize();
            ifs.read((char*)descriptors.data, data_size);
        }
        
        // Create Frame (dummy frame to construct KF)
        cv::Mat dummy_img;
        Frame::Ptr frame = std::make_shared<Frame>(id, timestamp, camera, dummy_img);
        frame->setPose(T_cw);
        frame->keypoints_ = keypoints;
        frame->descriptors_ = descriptors;
        
        Keyframe::Ptr kf = std::make_shared<Keyframe>(frame);
        
        // Associations (read but process later)
        std::vector<long long> lm_ids(num_kps);
        for (size_t j = 0; j < num_kps; ++j) {
            ifs.read((char*)&lm_ids[j], sizeof(long long));
        }
        kf_lm_ids[id] = lm_ids;
        
        map->addKeyframe(kf);
    }
    
    // 3. Load Landmarks
    size_t num_lms = 0;
    ifs.read((char*)&num_lms, sizeof(size_t));
    
    for (size_t i = 0; i < num_lms; ++i) {
        unsigned long id;
        ifs.read((char*)&id, sizeof(unsigned long));
        
        Vec3 pos;
        ifs.read((char*)pos.data(), sizeof(double) * 3);
        
        int rows, cols, type;
        ifs.read((char*)&rows, sizeof(int));
        ifs.read((char*)&cols, sizeof(int));
        ifs.read((char*)&type, sizeof(int));
        cv::Mat descriptor;
        if (rows > 0 && cols > 0) {
            descriptor.create(rows, cols, type);
            size_t data_size = descriptor.total() * descriptor.elemSize();
            ifs.read((char*)descriptor.data, data_size);
        }
        
        Landmark::Ptr lm = std::make_shared<Landmark>(id, pos);
        lm->descriptor_ = descriptor;
        
        map->addLandmark(lm);
    }
    
    // 4. Reconnect
    auto all_kfs = map->getAllKeyframes();
    auto all_lms = map->getAllLandmarks();
    
    for (auto& pair : all_kfs) {
        auto kf = pair.second;
        const auto& lm_ids = kf_lm_ids[kf->id_];
        
        // Resize landmarks vector in KF (it might be empty from constructor)
        kf->landmarks_.resize(lm_ids.size(), nullptr);
        
        for (size_t j = 0; j < lm_ids.size(); ++j) {
            long long lm_id = lm_ids[j];
            if (lm_id != -1) {
                if (all_lms.count(lm_id)) {
                    auto lm = all_lms[lm_id];
                    kf->landmarks_[j] = lm;
                    lm->addObservation(kf, j);
                }
            }
        }
        
        // Rebuild connections
        kf->updateConnections();
    }
    
    std::cout << "MapIO: Loaded " << num_kfs << " KFs and " << num_lms << " LMs." << std::endl;
    return true;
}

}
