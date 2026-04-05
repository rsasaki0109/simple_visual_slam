cmake_minimum_required(VERSION 3.14)
project(SimpleVisualSLAM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# Dependencies
find_package(OpenCV 4.5 REQUIRED)
find_package(CURL QUIET)  # Optional on Windows
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Ceres
# set(Ceres_DIR "/home/autoware/workspace/ceres-bin" CACHE PATH "Path to CeresConfig.cmake")
find_package(Ceres 2.1 QUIET)
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

    # Avoid depending on system glog/gflags.
    set(MINIGLOG ON CACHE BOOL "" FORCE)
    set(GFLAGS OFF CACHE BOOL "" FORCE)
    
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

# DBoW2 for loop closing
option(USE_DBOW2 "Enable DBoW2 for loop closing" ON)
if(USE_DBOW2)
    include(FetchContent)

    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Declare(
        DBoW2
        GIT_REPOSITORY https://github.com/dorian3d/DBoW2.git
        GIT_TAG master
        PATCH_COMMAND ${CMAKE_COMMAND} -E echo "Patching DBoW2..."
    )

    # Populate without configuring
    FetchContent_GetProperties(DBoW2)
    if(NOT dbow2_POPULATED)
        FetchContent_Populate(DBoW2)
    endif()

    # Patch DBoW2 CMakeLists.txt for compatibility BEFORE add_subdirectory
    file(READ "${dbow2_SOURCE_DIR}/CMakeLists.txt" dbow2_cmake)
    # Fix cmake_minimum_required for CMake 4.x
    string(REPLACE "cmake_minimum_required(VERSION 2.8)" "cmake_minimum_required(VERSION 3.14)" dbow2_cmake "${dbow2_cmake}")
    # Force STATIC lib
    string(REPLACE "set(LIB_SHARED \"SHARED\")" "set(LIB_SHARED \"STATIC\")" dbow2_cmake "${dbow2_cmake}")
    # Update C++ standard
    string(REPLACE "CXX_STANDARD 11" "CXX_STANDARD 17" dbow2_cmake "${dbow2_cmake}")
    file(WRITE "${dbow2_SOURCE_DIR}/CMakeLists.txt" "${dbow2_cmake}")

    # Patch TemplatedVocabulary.h to add loadFromTextFile (for ORB-SLAM vocab format)
    file(READ "${dbow2_SOURCE_DIR}/include/DBoW2/TemplatedVocabulary.h" _tv_h)
    if(NOT _tv_h MATCHES "loadFromTextFile")
        # Add method declaration after "void load(const std::string"
        string(REPLACE
            "void load(const std::string &filename);"
            "void load(const std::string &filename);\n  void loadFromTextFile(const std::string &filename);"
            _tv_h "${_tv_h}")
        # Add method implementation before the final #endif
        string(REPLACE
            "} // namespace DBoW2"
            "template<class TDescriptor, class F>\nvoid TemplatedVocabulary<TDescriptor,F>::loadFromTextFile(const std::string &filename)\n{\n  m_words.clear();\n  m_nodes.clear();\n  std::ifstream f(filename);\n  if(!f.is_open()) throw std::string(\"Could not open file \") + filename;\n  int n1, n2;\n  f >> m_k >> m_L >> n1 >> n2;\n  m_scoring = (ScoringType)n1;\n  m_weighting = (WeightingType)n2;\n  createScoringObject();\n  int expected_nodes = (int)((std::pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));\n  m_nodes.resize(expected_nodes);\n  m_nodes[0].id = 0;\n  for(int i = 1; i < expected_nodes; i++)\n  {\n    m_nodes[i].id = i;\n    int nIsLeaf;\n    f >> m_nodes[i].parent >> nIsLeaf;\n    std::stringstream ssd;\n    for(int iD = 0; iD < F::L; iD++) { std::string s; f >> s; ssd << s << \" \"; }\n    F::fromString(m_nodes[i].descriptor, ssd.str());\n    f >> m_nodes[i].weight;\n    m_nodes[m_nodes[i].parent].children.push_back(i);\n    if(nIsLeaf > 0)\n    {\n      int wid = (int)m_words.size();\n      m_words.push_back(&m_nodes[i]);\n      m_nodes[i].word_id = wid;\n    }\n  }\n}\n\n} // namespace DBoW2"
            _tv_h "${_tv_h}")
        file(WRITE "${dbow2_SOURCE_DIR}/include/DBoW2/TemplatedVocabulary.h" "${_tv_h}")
        message(STATUS "Patched DBoW2 with loadFromTextFile support")
    endif()

    # Now add the subdirectory
    add_subdirectory(${dbow2_SOURCE_DIR} ${dbow2_BINARY_DIR})

    add_definitions(-DUSE_DBOW2)
    message(STATUS "DBoW2 enabled via FetchContent")
else()
    message(STATUS "DBoW2 disabled - loop closing not available")
endif()

# Deep Learning Depth Estimation (ONNX Runtime)
option(USE_DEPTH_DL "Enable deep learning depth estimation (ONNX Runtime)" OFF)
if(USE_DEPTH_DL)
    # Try system-installed ONNX Runtime first
    find_library(ONNXRUNTIME_LIB onnxruntime)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime)

    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        message(STATUS "Found system ONNX Runtime: ${ONNXRUNTIME_LIB}")
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE}"
        )
    else()
        message(STATUS "ONNX Runtime not found, fetching pre-built release...")
        include(FetchContent)
        set(ONNXRUNTIME_VERSION "1.17.1")
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
        )
        FetchContent_MakeAvailable(onnxruntime)

        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        )
    endif()

    add_definitions(-DUSE_DEPTH_DL)
    message(STATUS "Deep learning depth estimation: ENABLED")
else()
    message(STATUS "Deep learning depth estimation: DISABLED")
endif()

# Core Library
add_library(svslam_reference_policy_core
    src/core/heuristic_reference_keyframe_policy.cc
)

target_include_directories(svslam_reference_policy_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

add_library(svslam_reference_policy_experiments
    src/experiments/reference_keyframe/score_reference_keyframe_policy.cc
    src/experiments/reference_keyframe/pipeline_reference_keyframe_policy.cc
)

target_include_directories(svslam_reference_policy_experiments PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(svslam_reference_policy_experiments PUBLIC
    svslam_reference_policy_core
)

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
)

target_link_libraries(svslam_core PUBLIC
    svslam_reference_policy_core
    ${OpenCV_LIBS}
    Ceres::ceres
    Sophus::Sophus
)

if(CURL_FOUND)
    target_link_libraries(svslam_core PUBLIC ${CURL_LIBRARIES})
endif()

if(USE_DBOW2 AND TARGET DBoW2)
    target_link_libraries(svslam_core PUBLIC DBoW2)
    target_include_directories(svslam_core PUBLIC ${dbow2_SOURCE_DIR}/include)
endif()

if(USE_DEPTH_DL)
    target_sources(svslam_core PRIVATE src/depth/onnx_depth_estimator.cc)
    target_link_libraries(svslam_core PUBLIC onnxruntime::onnxruntime)
endif()


# App
add_executable(run_mono apps/run_mono.cc)
target_link_libraries(run_mono PRIVATE svslam_core svslam_reference_policy_experiments)

add_executable(reference_policy_experiments
    tools/reference_policy_experiments.cc
)
target_link_libraries(reference_policy_experiments PRIVATE svslam_reference_policy_experiments)
target_include_directories(reference_policy_experiments PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Tests
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(svslam_tests
        tests/test_camera.cc
        tests/test_landmark.cc
        tests/test_map.cc
        tests/test_optimizer.cc
        tests/test_reference_keyframe_policy.cc
    )
    target_link_libraries(svslam_tests PRIVATE
        svslam_core
        svslam_reference_policy_experiments
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(svslam_tests)
endif()
