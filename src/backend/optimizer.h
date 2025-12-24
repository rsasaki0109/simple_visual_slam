#pragma once

#include "core/common.h"
#include "core/map.h"
#include "core/frame.h"
#include "core/keyframe.h"
#include "core/landmark.h"

namespace svslam {

class Optimizer {
public:
    // Local Bundle Adjustment
    // Optimize a keyframe and its neighbors, and observed landmarks
    static void bundleAdjustment(const std::vector<Keyframe::Ptr>& keyframes, 
                                 const std::vector<Landmark::Ptr>& landmarks,
                                 int iterations = 10);
                                 
    // Pose optimization only (e.g. for tracking)
    static int poseOptimization(Frame::Ptr frame);
};

}
