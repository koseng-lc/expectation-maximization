cmake_minimum_required(VERSION 2.8.1)
project(expectation_maximization)
add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} expectation_maximization.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
