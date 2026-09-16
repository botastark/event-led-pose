cmake_minimum_required(VERSION 3.16)

project(event_led_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    MetavisionSDK
    COMPONENTS core stream
    REQUIRED
)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

add_library(
    fast_event_viewer
    src/fast_event_viewer.cpp
)

target_include_directories(
    fast_event_viewer
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    fast_event_viewer
    PUBLIC
    OpenGL::GL
    GLEW::GLEW
    glfw
)

target_compile_options(
    fast_event_viewer
    PRIVATE
    -O3
    -march=native
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
    fast_event_viewer
    Threads::Threads

)

target_compile_options(
    live_frequency_ring
    PRIVATE
    -O3
    -march=native
)
