cmake_minimum_required(VERSION 3.16)

project(event_frequency_filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MetavisionSDK COMPONENTS core stream REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

add_executable(
    live_frequency_ring
    live_frequency_ring.cpp
)

target_link_libraries(
    live_frequency_ring
    PRIVATE
    MetavisionSDK::core
    MetavisionSDK::stream
    ${OpenCV_LIBS}
)

target_include_directories(
    live_frequency_ring
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(
    live_frequency_ring
    PRIVATE
    -O3
    -march=native
)
