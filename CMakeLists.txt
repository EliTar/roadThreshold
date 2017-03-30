cmake_minimum_required(VERSION 3.2)

project(roadthreshold)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(roadthreshold roadthreshold.cpp)
target_link_libraries(roadthreshold ${OpenCV_LIBS})

add_executable(thresholdfun thresholdfun.cpp)
target_link_libraries(thresholdfun ${OpenCV_LIBS})

