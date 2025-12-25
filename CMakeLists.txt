cmake_minimum_required(VERSION 3.14)
project(SimpleVisualSLAM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Dependencies
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4.5 REQUIRED) # Request specific version to avoid picking up 4.12 in /usr/local
find_package(CURL REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Ceres
# set(Ceres_DIR "/home/autoware/workspace/ceres-bin" CACHE PATH "Path to CeresConfig.cmake")
# find_package(Ceres REQUIRED)
message(STATUS "Ceres Solver disabled for Phase 1 (Data Structures & ORB Extraction). Enable for Optimization phases.")

# Sophus
find_package(Sophus QUIET)
if(NOT Sophus_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Sophus
        GIT_REPOSITORY https://github.com/strasdat/Sophus.git
        GIT_TAG 1.22.10 # MIT licensed version (older versions are GPL, but 1.22+ is MIT usually, check license)
        # Actually Sophus changed to MIT around 2017. 1.22.10 is safe.
    )
    FetchContent_MakeAvailable(Sophus)
endif()

# DBoW2 (Loop用)
if (NOT DBoW2_FOUND)
    # Fallback or placeholder if standard find fails, but for this task I will assume it's linked properly later.
    # For now, let's just proceed.
    message(STATUS "DBoW2 not found via find_package, assuming standard include/lib paths or user provided.")
endif()

# Core Library
add_library(svslam_core
    src/core/camera.cc
    src/core/frame.cc
    src/core/keyframe.cc
    src/core/landmark.cc
    src/core/map.cc
    src/io/map_io.cc
    src/tracking/tracking.cc
    src/tracking/initializer.cc
    src/backend/local_mapping.cc
)

target_include_directories(svslam_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4
)

target_link_libraries(svslam_core PUBLIC
    ${OpenCV_LIBS}
    opencv_videoio
    ${CURL_LIBRARIES}
    # ceres
    Sophus::Sophus
)

# App
add_executable(run_mono apps/run_mono.cc)
target_link_libraries(run_mono PRIVATE svslam_core)
