# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

cmake_minimum_required(VERSION 3.11)
project(ndarray LANGUAGES C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
OPTION(BUILD_TESTS "test suite" ON)
OPTION(BUILD_BENCHMARKS "benchmark suite" ON)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_EXTENSIONS        OFF)  # Force c++1[7z] instead of gnu++1[7z]
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # Fail if no c++17 compiler available

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory(${PROJECT_SOURCE_DIR}/ndarray/)
if(BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/test/)
endif()
if(BUILD_BENCHMARKS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark/)
endif()
