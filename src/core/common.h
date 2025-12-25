#pragma once

#include <memory>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <mutex>
#include <atomic>
#include <iostream>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace svslam {

using SE3 = Sophus::SE3d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using Mat33 = Eigen::Matrix3d;

// Forward declarations
class Frame;
class Keyframe;
class Landmark;
class Camera;
class Map;

}
