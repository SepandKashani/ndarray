# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

set(NDARRAY_BENCHMARKS
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_ndarray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_ndarray_operator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_ndcontainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_ndfunc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_nditer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_ndlinalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_ndutil.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp)  # main()
set(NDARRAY_TARGET benchmark_ndarray)
add_executable(${NDARRAY_TARGET} ${NDARRAY_BENCHMARKS})
target_link_libraries(${NDARRAY_TARGET} Ndarray)
