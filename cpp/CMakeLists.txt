cmake_minimum_required(VERSION 3.20)

project(event_led_classifier_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(MetavisionSDK COMPONENTS core stream REQUIRED)
add_executable(
    event_classifier_poc
    event_classifier_poc.cpp
    evk4_stream_probe evk4_stream_probe.cpp
    )

target_compile_options(
    event_classifier_poc
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    evk4_stream_probe
    PRIVATE
        MetavisionSDK::core
        MetavisionSDK::stream
)

include(CTest)

if(BUILD_TESTING)
    add_test(
        NAME synthetic_event_classifier
        COMMAND event_classifier_poc
    )
endif()