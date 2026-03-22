#pragma once

#include "core/common.h"
#include "io/tum_dataset.h"
#include <vector>
#include <cmath>

namespace svslam {

class AccelerometerProcessor {
public:
    // Estimate gravity direction from accelerometer readings during low-motion periods.
    // Returns normalized gravity vector in sensor frame.
    static Vec3 estimateGravity(const std::vector<AccelEntry>& measurements) {
        if (measurements.empty()) return Vec3(0, 0, 0);

        Vec3 sum(0, 0, 0);
        for (const auto& m : measurements) {
            sum += Vec3(m.ax, m.ay, m.az);
        }
        Vec3 mean = sum / static_cast<double>(measurements.size());
        double mag = mean.norm();

        // Sanity check: magnitude should be close to 9.81
        if (mag < 8.0 || mag > 12.0) return Vec3(0, 0, 0);

        return mean.normalized();
    }

    // Check if the sensor is approximately stationary (low acceleration variance).
    static bool isStationary(const std::vector<AccelEntry>& measurements, double threshold = 0.5) {
        if (measurements.size() < 5) return false;

        Vec3 sum(0, 0, 0);
        for (const auto& m : measurements) {
            sum += Vec3(m.ax, m.ay, m.az);
        }
        Vec3 mean = sum / static_cast<double>(measurements.size());

        double variance = 0.0;
        for (const auto& m : measurements) {
            Vec3 diff = Vec3(m.ax, m.ay, m.az) - mean;
            variance += diff.squaredNorm();
        }
        variance /= static_cast<double>(measurements.size());

        return variance < threshold;
    }

    // Compute rotation to align estimated gravity with world -Z axis.
    // Returns R_align such that R_align * gravity_sensor ≈ [0, 0, -1]
    static Mat33 computeGravityAlignment(const Vec3& gravity_direction) {
        Vec3 g = gravity_direction.normalized();
        Vec3 target(0, 0, -1);  // World gravity direction (TUM convention: Z-up)

        // If gravity is already aligned, return identity
        double dot = g.dot(target);
        if (std::abs(dot - 1.0) < 1e-6) {
            return Mat33::Identity();
        }
        if (std::abs(dot + 1.0) < 1e-6) {
            // 180 degree rotation around X axis
            Mat33 R;
            R << 1, 0, 0,
                 0, -1, 0,
                 0, 0, -1;
            return R;
        }

        // Rodrigues rotation
        Vec3 axis = g.cross(target);
        axis.normalize();
        double angle = std::acos(std::clamp(dot, -1.0, 1.0));

        Eigen::AngleAxisd aa(angle, axis);
        return aa.toRotationMatrix();
    }
};

}
