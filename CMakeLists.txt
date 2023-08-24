cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_BUILD_TYPE "Debug")

# add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
add_compile_options(-fPIC)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

project(my_engine)
option(BUILD_BACKEND_TENSORRT "Build tensorrt" ON)
option(BUILD_BACKEND_NCNN "Build ncnn"  ON)
option(BUILD_BACKEND_ONNXRUNTIME "Build onnxruntime"  ON)
option(NCNN_VULKAN "Build onnxruntime"  OFF)

add_subdirectory(deploy)