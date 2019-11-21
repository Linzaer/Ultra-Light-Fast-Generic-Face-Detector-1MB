cmake_minimum_required(VERSION 3.10)
project(Ultra-face-mnn)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(
        mnn/include
        src
)

link_directories(mnn/lib)

add_executable(Ultra-face-mnn src/main.cpp src/UltraFace.cpp)
target_link_libraries(Ultra-face-mnn MNN ${OpenCV_LIBS})