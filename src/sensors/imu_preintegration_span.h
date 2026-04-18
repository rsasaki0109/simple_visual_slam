#pragma once

#include <sophus/so3.hpp>

#include "core/common.h"

namespace svslam {

// Frozen preintegration between two consecutive keyframes. The holder (a
// Keyframe) owns the span FROM the previous KF TO itself, so `from_kf_id`
// points at the predecessor and the KF's own id is the target.
//
// delta_R / delta_v / delta_p are all expressed in the from-KF's IMU-body
// frame at the bias snapshot recorded in bias_accel / bias_gyro. Future
// bias updates can be applied linearly on top of the stored deltas when a
// VIO backend wires in bias Jacobians; Stage 0c keeps biases fixed at the
// values used for integration.
struct ImuPreintegrationSpan {
    Sophus::SO3d delta_R;              // rotation accumulated i->j in body frame
    Vec3 delta_v = Vec3::Zero();       // velocity delta in body-at-i frame
    Vec3 delta_p = Vec3::Zero();       // position delta in body-at-i frame
    double dt = 0.0;                   // integration duration in seconds
    Vec3 bias_accel = Vec3::Zero();    // accel bias used during integration
    Vec3 bias_gyro = Vec3::Zero();     // gyro bias used during integration
    unsigned long from_kf_id = 0;      // predecessor keyframe id
    bool valid = false;                // false until populated + sanity-checked

    // Camera-from-IMU rigid transform captured at span-creation time. Used
    // by BA to translate body-frame deltas into camera-frame pose
    // constraints. Identity when the dataset didn't provide an extrinsic.
    SE3 T_cam_imu;
};

}  // namespace svslam
