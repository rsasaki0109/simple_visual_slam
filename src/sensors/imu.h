#pragma once

#include "core/common.h"

namespace svslam {

// A single IMU measurement (accel + gyro) at a given timestamp.
// Units: m/s^2 for accel, rad/s for gyro. Timestamp in seconds.
struct ImuEntry {
    double timestamp_sec = 0.0;
    Vec3 accel = Vec3::Zero();
    Vec3 gyro = Vec3::Zero();
};

}  // namespace svslam
