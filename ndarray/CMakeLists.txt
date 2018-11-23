# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

# Make sure all definitions/flags/linked-libraries are automatically filled in
# if target(Ndarray::Ndarray) is used.

find_package(Eigen3    3.3.5  REQUIRED NO_MODULE)
find_package(MKL              REQUIRED           NO_SYSTEM_ENVIRONMENT_PATH)
find_package(OpenMP           REQUIRED)
find_package(FFTW3            REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)

add_library(Ndarray INTERFACE)
target_include_directories(Ndarray SYSTEM BEFORE INTERFACE ${FFTW3_INCLUDE_DIRS})
target_compile_definitions(Ndarray INTERFACE EIGEN_USE_MKL_ALL)
target_compile_options(Ndarray INTERFACE -Wall -Wextra -m64)
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    target_compile_options(Ndarray INTERFACE -g -Og)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
    target_compile_definitions(Ndarray INTERFACE NDEBUG) # disable (non-critical) safety checks
    target_compile_options(Ndarray INTERFACE -mfma -mavx -O3 -march=native)
endif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
target_link_libraries(Ndarray INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX MKL::MKL)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS Ndarray EXPORT Ndarray)
install(EXPORT Ndarray
        NAMESPACE Ndarray::
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/ndarray
        FILE NdarrayConfig.cmake)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
                       PATTERN "*.cpp")
