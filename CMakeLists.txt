# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

cmake_minimum_required(VERSION 3.11)
project(ndarray LANGUAGES C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")
OPTION(BUILD_TESTS "test suite" OFF)

set(CMAKE_CXX_STANDARD          14)
set(CMAKE_CXX_EXTENSIONS        OFF)  # Force c++1[4y] instead of gnu++1[4y]
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # Fail if no c++14 compiler available

add_subdirectory(${PROJECT_SOURCE_DIR}/src/)
if(BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/test/)
endif()
