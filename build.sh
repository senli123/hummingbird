#!bin/bash
# Program
#   build a c project
# History
# 2019/05/22        Lius        First release
test -e build/ && rm -rf ./build
test -e build/ || mkdir build
echo "mkdir build/"

#test -e bin/ && rm -rf ./bin
test -e bin/ || mkdir bin
echo "mkdir bin/"

cd build/
#cmake .. -DONNXRUNTIME_head_DIR=/workspace/lisen/_bushu/lisen/learn/onnxruntime/include/onnxruntime/core -DONNXRUNTIME_lib_DIR=/workspace/lisen/_bushu/lisen/learn/onnxruntime/build/Linux/RelWithDebInfo
cmake ..
make
