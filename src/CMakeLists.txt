# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

# ndarray is meant to be fast, hence we make sure all definitions/flags/
# linked-libraries are automatically filled in if target target(ndarray) is
# used.

find_package(Eigen3    3.3.5  REQUIRED NO_MODULE)
find_package(MKL              REQUIRED           NO_SYSTEM_ENVIRONMENT_PATH)
find_package(OpenMP           REQUIRED)
find_package(FFTW3            REQUIRED NO_MODULE NO_SYSTEM_ENVIRONMENT_PATH)

add_library(ndarray INTERFACE)
target_include_directories(ndarray SYSTEM BEFORE INTERFACE ${FFTW3_INCLUDE_DIRS})
target_compile_definitions(ndarray INTERFACE EIGEN_USE_MKL_ALL)
target_compile_options(ndarray INTERFACE -Wall -Wextra -m64)
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    target_compile_options(ndarray INTERFACE -g -Og)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
    target_compile_definitions(ndarray INTERFACE NDEBUG) # disable (non-critical) safety checks
    target_compile_options(ndarray INTERFACE -mfma -mavx -O3 -march=native)
endif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
target_link_libraries(ndarray INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX MKL::MKL)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(NDARRAY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndarray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndarray.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndcontainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndcontainer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndfunc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndfunc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_nditer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_nditer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndlinalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndlinalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndtype.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndutil.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/_ndutil.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ndarray.hpp)
install(FILES ${NDARRAY_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ndarray)
