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
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    calib3d
)

add_library(
    center_worker
    src/center_worker.cpp
)

target_include_directories(
    center_worker
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    center_worker
    PUBLIC
    Threads::Threads
)

target_compile_options(
    center_worker
    PRIVATE
    -O3
    -march=native
)

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
    center_worker
    OpenGL::GL
    GLEW::GLEW
    glfw
    Threads::Threads
)

target_compile_options(
    fast_event_viewer
    PRIVATE
    -O3
    -march=native
)

add_library(pose_estimator
    src/pose_estimator.cpp
)

target_include_directories(pose_estimator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pose_estimator
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(
    live_frequency_center
    src/live_frequency_center.cpp
)

target_include_directories(
    live_frequency_center
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    live_frequency_center
    PRIVATE
    MetavisionSDK::core
    MetavisionSDK::stream
    fast_event_viewer
    center_worker
    pose_estimator
)

target_compile_options(
    live_frequency_center
    PRIVATE
    -O3
    -march=native
)
