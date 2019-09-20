# #############################################################################
# CMakeLists.txt
# ==============
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

add_library(Ndarray INTERFACE)
target_compile_options(Ndarray INTERFACE -Wall -Wextra -m64)
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    target_compile_options(Ndarray INTERFACE -g -Og)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
    target_compile_definitions(Ndarray INTERFACE NDEBUG) # disable (non-critical) safety checks
    target_compile_options(Ndarray INTERFACE -mfma -mavx -O3 -march=native)
endif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
target_link_libraries(Ndarray INTERFACE ${CONAN_LIBS})

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
