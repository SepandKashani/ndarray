# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

set(NDARRAY_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndarray_operator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndcontainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndfunc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_nditer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndlinalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ndutil.cpp)

set(NDARRAY_TARGET test_ndarray)

add_executable(${NDARRAY_TARGET} ${NDARRAY_TESTS})
target_link_libraries(${NDARRAY_TARGET} Ndarray)
