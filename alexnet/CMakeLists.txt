add_executable(alexnet ${CMAKE_CURRENT_SOURCE_DIR}/alexnet.cpp)
target_link_libraries(alexnet nvinfer)
target_link_libraries(alexnet cudart)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)