# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

find_package(GTest 1.8 REQUIRED)
find_package(Ndarray REQUIRED)

set(NDARRAY_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_constructor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_manipulation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_operator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_property.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_linalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndcontainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndutil.cpp)

set(NDARRAY_TARGET test_ndarray)

add_executable(${NDARRAY_TARGET} ${NDARRAY_TESTS})
target_link_libraries(${NDARRAY_TARGET} gtest gtest_main Ndarray::Ndarray)
