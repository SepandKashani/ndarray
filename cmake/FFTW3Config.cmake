# #############################################################################
# FFTW3Config.cmake
# =================
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

# Find FFTW3 Libraries
#
# This module defines the following variables:
#
#    FFTW3_FOUND          - True if the library was found
#    FFTW3_INCLUDE_DIRS   - FFTW3 header file directory.
#    FFTW3_LIBRARIES      - FFTW3 libraries to link against.

include(FindPackageHandleStandardArgs)

find_path(FFTW3_INCLUDE_DIR fftw3.h)
if(FFTW3_INCLUDE_DIR-NOTFOUND)
    message(FATAL_ERROR "Could not locate fftw3.h")
endif()

find_library(FFTW3_LIB_FLOAT fftw3f)
if(FFTW3_LIB_FLOAT-NOTFOUND)
    message(FATAL_ERROR "Could not locate library fftw3f")
endif()

find_library(FFTW3_LIB_FLOAT_OMP fftw3f_omp)
if(FFTW3_LIB_FLOAT_OMP-NOTFOUND)
    message(FATAL_ERROR "Could not locate library fftw3f_omp")
endif()

find_library(FFTW3_LIB_DOUBLE fftw3)
if(FFTW3_LIB_DOUBLE-NOTFOUND)
    message(FATAL_ERROR "Could not locate library fftw3")
endif()

find_library(FFTW3_LIB_DOUBLE_OMP fftw3_omp)
if(FFTW3_LIB_DOUBLE_OMP-NOTFOUND)
    message(FATAL_ERROR "Could not locate library fftw3_omp")
endif()

find_library(FFTW3_LIB_LONG fftw3l)
if(FFTW3_LIB_LONG-NOTFOUND)
    message(FATAL_ERROR "Could not locate library fftw3l")
endif()

find_library(FFTW3_LIB_LONG_OMP fftw3l_omp)
if(FFTW3_LIB_LONG_OMP-NOTFOUND)
    message(FATAL_ERROR "Could not locate library fftw3l_omp")
endif()

set(FFTW3_LIBRARY ${FFTW3_LIB_FLOAT} ${FFTW3_LIB_FLOAT_OMP} ${FFTW3_LIB_DOUBLE} ${FFTW3_LIB_DOUBLE_OMP} ${FFTW3_LIB_LONG} ${FFTW3_LIB_LONG_OMP})
find_package_handle_standard_args(FFTW3 DEFAULT_MSG
                                        FFTW3_INCLUDE_DIR
                                        FFTW3_LIBRARY)

if(FFTW3_FOUND)
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
endif()
