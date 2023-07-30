#不同backend支持的各种转换格式
torch2onnx = "torch2onnx"
torch2torchscript = "torch2torchscript"
torch2wts = "torch2wts"
BACKEND_IR_DICT = { 'TensorrtWrapper' : [torch2onnx, torch2wts],
                    'NcnnWrapper':[torch2onnx,torch2torchscript]}