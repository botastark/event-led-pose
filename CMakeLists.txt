cmake_minimum_required(VERSION 3.16)

project(event_led_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    MetavisionSDK
    COMPONENTS core stream ui
    REQUIRED
)

add_executable(
    live_frequency_ring
    src/live_frequency_ring.cpp
)

target_link_libraries(
    live_frequency_ring
    PRIVATE
    MetavisionSDK::core
    MetavisionSDK::stream
    MetavisionSDK::ui
)

target_compile_options(
    live_frequency_ring
    PRIVATE
    -O3
    -march=native
)
# additional executable for stream probe
add_executable(
    evk4_stream_probe
    src/evk4_stream_probe.cpp
)

target_link_libraries(
    evk4_stream_probe
    PRIVATE
    MetavisionSDK::core
    MetavisionSDK::stream
)