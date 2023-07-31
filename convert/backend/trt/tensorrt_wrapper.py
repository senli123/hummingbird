from __future__ import print_function
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from .common import *
from backend.base_backend import BaseBackend
class TensorrtWrapper(BaseBackend):
    def __init__(self,engine_path, cuda_id,input_h, 
                 input_w,output_size,input_name,output_name):
        self.cfx = cuda.Device(cuda_id).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            buf = f.read()
            self.engine = runtime.deserialize_cuda_engine(buf)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def infer(self,input):
        self.cfx.push()
        self.inputs[0].host = input
        trt_outputs = do_inference(self.context, bindings=self.bindings, 
                        inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        self.destory()
        return trt_outputs[0]
    def destory(self):
        self.cfx.pop()
    def get_name(self):
        return "TensorrtWrapper"

#         export CPATH=$CPATH:/usr/local/cuda/include
# export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
# pip install pycuda
#conda install -c conda-forge pycuda

    