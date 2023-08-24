#添加工具类
file(GLOB_RECURSE SRC_UTILS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)
file(GLOB_RECURSE HDR_INTERFACE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
)
list(APPEND SRCS ${SRC_UTILS_FILES})
list(APPEND HDRS ${HDR_INTERFACE_FILES})

#添加preprocess
file(GLOB_RECURSE SRC_PREPROCESS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/*.cpp
)
file(GLOB_RECURSE HDR_PREPROCESS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/*.h
)
list(APPEND SRCS ${SRC_PREPROCESS_FILES})
list(APPEND HDRS ${HDR_PREPROCESS_FILES})

#添加postprocess
file(GLOB_RECURSE SRC_POSTPROCESS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/postprocess/*.cpp
)
file(GLOB_RECURSE HDR_POSTPROCESS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/postprocess/*.h
)
list(APPEND SRCS ${SRC_POSTPROCESS_FILES})
list(APPEND HDRS ${HDR_POSTPROCESS_FILES})

#根据不同的backend进行添加
list(APPEND HDRS ${CMAKE_CURRENT_SOURCE_DIR}/backend/backend_interface.h)

if(BUILD_BACKEND_TENSORRT)
    list(APPEND HDRS ${CMAKE_CURRENT_SOURCE_DIR}/backend/logging.h)
    list(APPEND SRCS ${CMAKE_CURRENT_SOURCE_DIR}/backend/tensorrt_wrapper.cpp)
    list(APPEND HDRS ${CMAKE_CURRENT_SOURCE_DIR}/backend/tensorrt_wrapper.h)
endif()

if(BUILD_BACKEND_NCNN)
    list(APPEND SRCS ${CMAKE_CURRENT_SOURCE_DIR}/backend/ncnn_wrapper.cpp)
    list(APPEND HDRS ${CMAKE_CURRENT_SOURCE_DIR}/backend/ncnn_wrapper.h)
endif()

if (BUILD_BACKEND_ONNXRUNTIME)
list(APPEND SRCS ${CMAKE_CURRENT_SOURCE_DIR}/backend/ort_wrapper.cpp)
list(APPEND HDRS ${CMAKE_CURRENT_SOURCE_DIR}/backend/ort_wrapper.h)
endif()

#添加service
file(GLOB_RECURSE SRC_SERVICE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/service/*.cpp
)
file(GLOB_RECURSE HDR_SERVICE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/service/*.h
)
list(APPEND SRCS ${SRC_SERVICE_FILES})
list(APPEND HDRS ${HDR_SERVICE_FILES})

add_library(Engine SHARED ${SRCS})
target_include_directories(Engine PUBLIC ${HDRS})

#根据不同的backend链接不同的库文件
if(BUILD_BACKEND_ONNXRUNTIME)
set(ONNXRUNTIME_head_DIR /workspace/lisen/tensorrt/my_tensorrt/third_backend/onnxruntime-linux-x64-gpu-1.15.1/include)
    set(ONNXRUNTIME_lib_DIR /workspace/lisen/tensorrt/my_tensorrt/third_backend/onnxruntime-linux-x64-gpu-1.15.1/lib)
    message(STATUS "ONNXRUNTIME_head_DIR: ${ONNXRUNTIME_head_DIR}")
    message(STATUS "ONNXRUNTIME_lib_DIR: ${ONNXRUNTIME_lib_DIR}")
    target_include_directories(Engine PUBLIC "${ONNXRUNTIME_head_DIR}")
    target_link_libraries(Engine PUBLIC "${ONNXRUNTIME_lib_DIR}/libonnxruntime.so")
endif()

if(BUILD_BACKEND_TENSORRT)
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    # TODO(Call for PR): make cmake compatible with Windows
    # set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    # enable_language(CUDA)
    # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
    # cuda
    target_include_directories(Engine PUBLIC /usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    # tensorrt
    # TODO(Call for PR): make TRT path configurable from command line
    target_include_directories(Engine PUBLIC /usr/include/x86_64-linux-gnu/)
    link_directories(/usr/lib/x86_64-linux-gnu/)
    target_link_libraries(Engine PUBLIC nvinfer)
    target_link_libraries(Engine PUBLIC /usr/local/cuda/lib64/libcudart.so.12)
endif()

if(BUILD_BACKEND_NCNN)
# set(NCNN_head_DIR /workspace/lisen/tensorrt/my_tensorrt/third_backend/ncnn/build/install/include)
#     set(NCNN_lib_DIR /workspace/lisen/tensorrt/my_tensorrt/third_backend/ncnn/build/install/lib)
    set(NCNN_head_DIR /workspace/lisen/tensorrt/my_tensorrt/third_backend/ncnn/build/install/include)
    set(NCNN_lib_DIR /workspace/lisen/tensorrt/my_tensorrt/third_backend/ncnn/build/install/lib)
    
    message(STATUS "NCNN_head_DIR: ${NCNN_head_DIR}")
    message(STATUS "NCNN_lib_DIR: ${NCNN_lib_DIR}")
    target_include_directories(Engine PUBLIC "${NCNN_head_DIR}")
    target_link_libraries(Engine PUBLIC "${NCNN_lib_DIR}/libncnnd.so.1.0.20230824")
endif()
# opencv
find_package(OpenCV)
message(STATUS "Find opencv include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find opencv lib at ${OpenCV_LIBRARIES}")
target_include_directories(Engine PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Engine PUBLIC ${OpenCV_LIBRARIES})

#--------------------json---------------------------------
set(json_INCLUDE_DIR "/workspace/lisen/tensorrt/my_tensorrt/third/jsoncpp-src-0.5.0/include")
set(json_LIBRARIES "/workspace/lisen/tensorrt/my_tensorrt/third/jsoncpp-src-0.5.0/libs/linux-gcc-5.4.0/libjson_linux-gcc-5.4.0_libmt.so")

#----------------cv-cuda----------------
set(CVCUDA_INCLUDE_DIR "/workspace/lisen/package/opt/nvidia/cvcuda0/include")
set(CVCUDA_LIBRARIES "/workspace/lisen/package/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu")

target_include_directories(Engine PUBLIC ${json_INCLUDE_DIR} ${CVCUDA_INCLUDE_DIR})
target_link_libraries(Engine PUBLIC ${json_LIBRARIES} ${CVCUDA_LIBRARIES})

include_directories( /workspace/lisen/tensorrt/my_tensorrt/CV-CUDA/samples)
add_executable(demo ${CMAKE_CURRENT_SOURCE_DIR}/demo/test.cpp)
target_link_libraries(demo PUBLIC Engine)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)





