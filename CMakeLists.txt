cmake_minimum_required(VERSION 3.14)
project(SimpleVisualSLAM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# Dependencies
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4.5 REQUIRED) # Request specific version to avoid picking up 4.12 in /usr/local
find_package(CURL REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Ceres
# set(Ceres_DIR "/home/autoware/workspace/ceres-bin" CACHE PATH "Path to CeresConfig.cmake")
find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    message(STATUS "Ceres not found, fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
        GIT_TAG 2.1.0
    )
    # Configure Ceres options to be lightweight
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
    set(PROVISION_DEPENDENCIES ON CACHE BOOL "" FORCE) # Let Ceres manage deps if needed

    # Avoid depending on system glog/gflags (which may pull in libfmt.so.9)
    set(CERES_USE_MINIGLOG ON CACHE BOOL "" FORCE)
    set(CERES_USE_GLOG OFF CACHE BOOL "" FORCE)
    set(CERES_USE_GFLAGS OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(ceres)
    
    # Create alias if needed
    if(NOT TARGET Ceres::ceres)
        add_library(Ceres::ceres ALIAS ceres)
    endif()
endif()


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

# DBoW2
find_package(DBoW2 QUIET)
if(NOT DBoW2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        DBoW2
        GIT_REPOSITORY https://github.com/dorian3d/DBoW2.git
        GIT_TAG master # Use master or a specific tag
    )
    
    FetchContent_GetProperties(DBoW2)
    if(NOT dbow2_POPULATED)
        FetchContent_Populate(DBoW2)
        
        # Patch DBoW2 CMakeLists.txt to force STATIC lib and C++17
        file(READ "${dbow2_SOURCE_DIR}/CMakeLists.txt" dbow2_cmake)
        # Use regex to replace set(LIB_SHARED "SHARED") with set(LIB_SHARED "STATIC")
        string(REGEX REPLACE "set\\(LIB_SHARED \"SHARED\"\\)" "set(LIB_SHARED \"STATIC\")" dbow2_cmake "${dbow2_cmake}")
        # Replace standard 11 with 17
        string(REGEX REPLACE "CXX_STANDARD 11" "CXX_STANDARD 17" dbow2_cmake "${dbow2_cmake}")
        file(WRITE "${dbow2_SOURCE_DIR}/CMakeLists.txt" "${dbow2_cmake}")
        
        add_subdirectory(${dbow2_SOURCE_DIR} ${dbow2_BINARY_DIR})
    endif()
endif()

# Core Library
add_library(svslam_core
    src/core/camera.cc
    src/core/frame.cc
    src/core/keyframe.cc
    src/core/landmark.cc
    src/core/map.cc
    src/io/euroc_dataset.cc
    src/io/tum_dataset.cc
    src/io/map_io.cc
    src/tracking/tracking.cc
    src/tracking/initializer.cc
    src/backend/local_mapping.cc
    src/backend/optimizer.cc
    src/loop_closing/loop_closing.cc
)

target_include_directories(svslam_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4
    ${dbow2_SOURCE_DIR}/include
)

target_link_libraries(svslam_core PUBLIC
    ${OpenCV_LIBS}
    opencv_videoio
    ${CURL_LIBRARIES}
    Ceres::ceres
    Sophus::Sophus
)

if(TARGET DBoW2)
    target_link_libraries(svslam_core PUBLIC DBoW2)
endif()

# App
add_executable(run_mono apps/run_mono.cc)
target_link_libraries(run_mono PRIVATE svslam_core)
if(TARGET DBoW2)
    target_link_libraries(run_mono PRIVATE DBoW2)
endif()
