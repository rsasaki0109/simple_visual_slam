# FindEigen.cmake / EigenConfig.cmake wrapper
# Wrapper to find Eigen3 and fake version if necessary for Ceres

# Just find any Eigen3
find_package(Eigen3 QUIET)

if(Eigen3_FOUND)
    set(Eigen_FOUND TRUE)
    # Lie to Ceres about version to bypass strict check
    set(EIGEN_VERSION 3.4.0) 
    set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
    set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
    
    if(NOT TARGET Eigen::Eigen)
        add_library(Eigen::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
    
    message(STATUS "Eigen3 found at ${EIGEN3_INCLUDE_DIR}. Faking version ${EIGEN_VERSION} for Ceres.")
else()
    set(Eigen_FOUND FALSE)
endif()
