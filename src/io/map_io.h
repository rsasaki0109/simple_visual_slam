#pragma once

#include "core/map.h"
#include <string>

namespace svslam {

class MapIO {
public:
    // Simple serialization to a binary or text file
    // For this task, we will just declare the interface.
    // The user requirement says: "Keyframe/Landmark/Camera/観測関係を永続化し再開可能"
    
    static bool saveMap(const std::string& filename, Map::Ptr map);
    static bool loadMap(const std::string& filename, Map::Ptr map);
};

}
