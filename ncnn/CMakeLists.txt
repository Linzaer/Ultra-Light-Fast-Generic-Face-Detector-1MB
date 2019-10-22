cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

project(Ultra-Light-Fast-Generic-Face-Detector-1MB)

add_subdirectory(3rdparty/ncnn ncnn_build)
include_directories(3rdparty/ncnn/src)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp src/UltraFace.cpp)
target_link_libraries(main ncnn ${OpenCV_LIBS})
